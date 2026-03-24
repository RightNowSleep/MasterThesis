"""
eval/entropy.py
---------------
Attention-entropy evaluator for LLaMA-style models.

Interface mirrors PerplexityEvaluator: same constructor style, same CLI
argument helpers (add_args_entropy / add_args_model), same generate_save_filename.

Measured quantity
-----------------
For every forward pass the model returns one causal attention-weight matrix
per layer (already softmax-ed, causally masked):

    A[l]  ∈  ℝ^{B × H × T_q × T_k}

Two entropy variants are computed at every (layer l, head h, position t):

    Raw Shannon entropy:
        H(l,h,t) = -∑_{k=0}^{t} A[l,h,t,k] · log(A[l,h,t,k] + ε)   [nats]

    Normalised entropy (position-adjusted, ∈ [0,1]):
        H_norm(l,h,t) = H(l,h,t) / log(t + 1 + ε)

    The normalised form corrects for the fact that position t can attend to
    at most (t+1) tokens, so its theoretical maximum entropy grows with t.
    H_norm = 1 ⟹ perfectly uniform; H_norm = 0 ⟹ delta-function attention.

Stored metrics (per evaluated sequence length)
----------------------------------------------
  entropy_layer_position          [L][T]      raw,  mean over H & samples
  norm_entropy_layer_position     [L][T]      norm, mean over H & samples
  entropy_head_layer              [L][H]      raw,  mean over all T & samples
  norm_entropy_head_layer         [L][H]      norm, mean over all T & samples
  entropy_head_layer_position     [L][H][T]   raw,  mean over samples only
  head_norm_std_by_layer          [L]         std of per-head H_norm across H
  raw_entropy_quartiles_by_layer  [L][5]      (min,Q1,med,Q3,max) of raw H
  norm_entropy_quartiles_by_layer [L][5]      same for normalised H
  top_k_concentration             [T]         top-k attn mass fraction
  top_k_boundary                  int         first position with window > k
"""

import gc
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import argparse

import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from models.model_loader import load_model, load_tokenizer, add_args_model

_EPS = 1e-9
_PREC = 5  # decimal places written to JSON


# ---------------------------------------------------------------------------
# Functional helpers
# ---------------------------------------------------------------------------


def _shannon_entropy(attn: torch.Tensor) -> torch.Tensor:
    """
    Shannon entropy in nats along the last (key) axis.

    Parameters
    ----------
    attn : Tensor  [..., T_q, T_k]   softmax weights, rows sum to 1

    Returns
    -------
    Tensor  [..., T_q]
    """
    return -(attn * (attn + _EPS).log()).sum(dim=-1)


def _position_normalise(ent: torch.Tensor) -> torch.Tensor:
    """
    Map raw entropy at position t to [0, 1] by dividing by log(t+1).

        H_norm(t) = H(t) / log(t + 1 + ε)

    The denominator is clamped to 1e-6 so position 0 never causes NaN.
    Positions 0 .. top_k-1 will appear near H_norm ≈ 1 regardless of model
    behaviour (tiny causal window); these are flagged in the JSON and
    visually annotated in the plot script.

    Parameters
    ----------
    ent : Tensor  [..., T]

    Returns
    -------
    Tensor  [..., T]  ∈ [0, 1]
    """
    T = ent.shape[-1]
    pos = torch.arange(T, dtype=ent.dtype, device=ent.device)
    den = (pos + 1.0).log().clamp(min=1e-6)
    return ent / den


# ---------------------------------------------------------------------------
# EntropyEvaluator
# ---------------------------------------------------------------------------


class EntropyEvaluator:
    r"""
    Attention-entropy evaluator for LLaMA models.

    Constructor parameters mirror PerplexityEvaluator so the two can be
    swapped in ``test.py`` with minimal changes.

    Attributes
    ----------
    model             : LlamaForCausalLM   (eval mode)
    tokenizer         : PreTrainedTokenizer
    dataset_path      : str
    split             : str
    limit             : int | None
    device            : str
    add_start_token   : bool
    max_length        : int
    min_length        : int
    length_step       : int | None
    top_k             : int
    aggressive_memory : bool
    length_list       : list[int]
    save_dir          : str
    save_file         : str
    """

    def __init__(
        self,
        model,
        tokenizer,
        dataset: str = "emozilla/proofpile-test-tokenized",
        split: str = "test",
        limit: int = 20,
        device: str = None,
        add_start_token: bool = True,
        max_length: int = 2048,
        min_length: int = 256,
        length_step: int = None,
        top_k: int = 10,
        aggressive_memory: bool = True,
        save_dir: str = "results/entropy",
        save_file: str = None,
    ):
        r"""
        Parameters
        ----------
        model             : pretrained LlamaForCausalLM
        tokenizer         : matching tokenizer
        dataset           : HuggingFace dataset path (needs 'input_ids' column)
        split             : dataset split
        limit             : sample cap (None → all)
        device            : 'cuda' | 'cpu' | 'gpu' | None (auto)
        add_start_token   : prepend BOS; tokenizer must have bos_token
        max_length        : upper bound of the evaluated length list
        min_length        : lower bound / first entry
        length_step       : fixed step; None → exponential ×2
        top_k             : k for the top-k concentration metric
        aggressive_memory : free GPU cache after every sample
        save_dir          : directory for JSON output
        save_file         : JSON filename (auto-generated when None)
        """
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.dataset_path = dataset
        self.split = split
        self.limit = limit
        self.add_start_token = add_start_token
        self.max_length = max_length
        self.min_length = min_length
        self.length_step = length_step
        self.top_k = top_k
        self.aggressive_memory = aggressive_memory

        # device
        if device is not None:
            assert device in (
                "gpu",
                "cpu",
                "cuda",
            ), "device must be 'gpu', 'cpu', or 'cuda'."
            self.device = "cuda" if device == "gpu" else device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # BOS check
        if self.add_start_token:
            assert self.tokenizer.bos_token is not None, (
                "add_start_token=True requires a BOS token in the tokenizer. "
                "Pass add_start_token=False or use a different model."
            )

        # dataset
        self.dataset = load_dataset(self.dataset_path, split=self.split)
        print(f"Dataset size (before limit): {len(self.dataset)}")
        if self.limit is not None:
            self.dataset = self.dataset[: self.limit]
        n_loaded = len(self.dataset["input_ids"])
        print(f"Effective samples          : {n_loaded}")

        # length schedule
        self.length_list = self._generate_length_list()
        print(f"Lengths to evaluate        : {self.length_list}")

        # output
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_file = save_file or f"{time.strftime('%Y%m%d-%H%M%S')}.json"

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _generate_length_list(self) -> list:
        """Exponential doubling or fixed step."""
        if self.length_step is not None:
            return list(range(self.min_length, self.max_length + 1, self.length_step))
        lst = [self.min_length]
        while True:
            nxt = lst[-1] * 2
            if nxt <= self.max_length:
                lst.append(nxt)
            else:
                break
        if self.max_length not in lst:
            lst.append(self.max_length)
        return lst

    # ------------------------------------------------------------------
    # core computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_entropy_at_length(self, max_length: int) -> dict:
        """
        Accumulate attention-entropy statistics over all samples at one length.

        Accumulation
        ^^^^^^^^^^^^
        All tensors are accumulated in float64 on CPU to prevent rounding
        errors from compounding over many samples.  The 4-bit quantised model
        produces bfloat16 attention weights; casting to float64 before any
        log/sum operations is essential for numerical correctness.

        Padding
        ^^^^^^^
        All samples are pre-truncated to ``max_tokenized_len`` tokens.  Samples
        shorter than this are zero-padded in the accumulator (entropy = 0 at
        padded positions), which is the neutral contribution.  The dataset used
        (proofpile-tokenized) typically supplies sequences longer than 2048.
        """
        encoded = self.dataset["input_ids"]
        masks = self.dataset["attention_mask"]

        max_tok = (max_length - 1) if self.add_start_token else max_length
        encoded = [x[:max_tok] for x in encoded]
        masks = [x[:max_tok] for x in masks]

        # Running accumulators – lazily initialised after the first forward pass
        acc_raw = None  # [L, H, max_length]  float64
        acc_norm = None  # [L, H, max_length]  float64
        acc_topk = None  # [max_length]         float64
        n = 0

        pbar = tqdm(
            total=len(encoded),
            desc=f"  seq_len={max_length}",
            leave=False,
        )

        for idx in range(len(encoded)):
            # ── construct input tensors ───────────────────────────────
            lbl = torch.tensor(encoded[idx : idx + 1])  # [1, T]
            mask = torch.tensor(masks[idx : idx + 1])  # [1, T]

            if self.add_start_token:
                bos = torch.tensor([[self.tokenizer.bos_token_id]])
                lbl = torch.cat([bos, lbl], dim=1)
                mask = torch.cat([torch.ones((1, 1), dtype=mask.dtype), mask], dim=1)

            ids = lbl.to(self.device)
            mask = mask.to(self.device)
            T = ids.size(1)

            # ── forward pass ─────────────────────────────────────────
            outputs = self.model(ids, attention_mask=mask, output_attentions=True)
            attentions = outputs.attentions  # tuple len=L, each [1,H,T,T]
            L = len(attentions)
            H = attentions[0].size(1)

            # ── lazy init ────────────────────────────────────────────
            if acc_raw is None:
                acc_raw = torch.zeros(L, H, max_length, dtype=torch.float64)
                acc_norm = torch.zeros(L, H, max_length, dtype=torch.float64)
                acc_topk = torch.zeros(max_length, dtype=torch.float64)

            topk_layer_sum = torch.zeros(max_length, dtype=torch.float64)

            for l_idx, attn in enumerate(attentions):
                # cast to float64 – critical for 4-bit quantised models
                a = attn[0].double().cpu()  # [H, T, T]
                ent = _shannon_entropy(a)  # [H, T]
                nrm = _position_normalise(ent)  # [H, T]

                # right-pad with zeros when actual seq < max_length
                if T < max_length:
                    pad = torch.zeros(
                        H,
                        max_length - T,
                        dtype=torch.float64,
                        device=ent.device,
                    )
                    ent = torch.cat([ent, pad], dim=1)
                    nrm = torch.cat([nrm, pad], dim=1)

                acc_raw[l_idx] += ent.cpu()
                acc_norm[l_idx] += nrm.cpu()

                # top-k concentration for this layer: mean over heads → [T]
                k = min(self.top_k, a.size(-1))
                tkv = a.topk(k, dim=-1).values.sum(-1).mean(0)  # [T]
                if T < max_length:
                    pad_tkv = torch.zeros(
                        max_length - T,
                        dtype=torch.float64,
                        device=tkv.device,
                    )
                    tkv = torch.cat([tkv, pad_tkv])
                topk_layer_sum += tkv.cpu().double()

            acc_topk += topk_layer_sum / L
            n += 1

            del outputs, attentions, ids, mask
            if self.aggressive_memory:
                gc.collect()
                torch.cuda.empty_cache()

            pbar.update(1)

        pbar.close()

        if n == 0:
            raise RuntimeError("No samples were processed.")

        # ── average over samples ──────────────────────────────────────
        avg_raw = (acc_raw / n).numpy()  # [L, H, max_length]
        avg_norm = (acc_norm / n).numpy()  # [L, H, max_length]
        avg_topk = (acc_topk / n).numpy()  # [max_length]

        # ------------------------------------------------------------------
        # Compute and package all output metrics
        # ------------------------------------------------------------------

        # 1 & 2: [L, T]  – mean over H; every position preserved
        elp = np.round(avg_raw.mean(axis=1), _PREC).tolist()
        nelp = np.round(avg_norm.mean(axis=1), _PREC).tolist()

        # 3 & 4: [L, H]  – mean over T
        #   raw version: averaging over T mixes different positional scales
        #   norm version: all positions are in [0,1], mean is statistically valid
        ehl = np.round(avg_raw.mean(axis=2), _PREC).tolist()
        nehl = np.round(avg_norm.mean(axis=2), _PREC).tolist()

        # 5: [L, H, T]  – full raw tensor (largest field, ~2 M floats at T=2048)
        ehlp = np.round(avg_raw, _PREC).tolist()

        # 6: [L]  – intra-layer head specialisation
        #    step 1: per-head mean of H_norm over T  →  [L, H]
        #    step 2: std across H                    →  [L]
        hstd = np.round(avg_norm.mean(axis=2).std(axis=1), _PREC).tolist()

        # 7 & 8: [L, 5]  – quartile distributions over H×T
        raw_q, nrm_q = [], []
        for l in range(avg_raw.shape[0]):
            raw_q.append(
                np.round(
                    np.quantile(avg_raw[l].ravel(), [0.0, 0.25, 0.5, 0.75, 1.0]),
                    _PREC,
                ).tolist()
            )
            nrm_q.append(
                np.round(
                    np.quantile(avg_norm[l].ravel(), [0.0, 0.25, 0.5, 0.75, 1.0]),
                    _PREC,
                ).tolist()
            )

        return {
            "entropy_layer_position": elp,
            "norm_entropy_layer_position": nelp,
            "entropy_head_layer": ehl,
            "norm_entropy_head_layer": nehl,
            "entropy_head_layer_position": ehlp,
            "head_norm_std_by_layer": hstd,
            "raw_entropy_quartiles_by_layer": raw_q,
            "norm_entropy_quartiles_by_layer": nrm_q,
            "top_k_concentration": np.round(avg_topk, _PREC).tolist(),
            "top_k_boundary": int(min(self.top_k, max_length)) - 1,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self) -> dict:
        r"""
        Evaluate at every length in ``self.length_list`` and write JSON.

        Returns
        -------
        dict   full output including metadata and per-length results
        """
        results = {}

        outer = tqdm(self.length_list, desc="EntropyEvaluator")
        for seq_len in outer:
            outer.set_postfix(seq_len=seq_len)
            torch.cuda.empty_cache()
            results[str(seq_len)] = self._compute_entropy_at_length(seq_len)

        first = results[str(self.length_list[0])]
        num_layers = len(first["entropy_layer_position"])
        num_heads = len(first["entropy_head_layer"][0])

        output = {
            "lengths": self.length_list,
            "model_name": getattr(self.model.config, "_name_or_path", "unknown"),
            "top_k": self.top_k,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "results": results,
        }

        save_path = os.path.join(self.save_dir, self.save_file)
        with open(save_path, "w") as f:
            json.dump(output, f, separators=(",", ":"))
        print(f"\nResults saved → {save_path}")

        # console summary
        print("\n  seq_len │ mean H (nats) │ mean H_norm")
        print("  ────────┼───────────────┼────────────")
        for sl in self.length_list:
            r = results[str(sl)]
            mh = float(np.mean(r["entropy_layer_position"]))
            mn = float(np.mean(r["norm_entropy_layer_position"]))
            print(f"  {sl:>7} │ {mh:>13.4f} │ {mn:.4f}")

        return output


# ---------------------------------------------------------------------------
# Argument helpers  (mirror PerplexityEvaluator convention)
# ---------------------------------------------------------------------------


def add_args_entropy(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register entropy-evaluation CLI arguments."""
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="emozilla/proofpile-test-tokenized",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--add-start-token", type=bool, default=True)
    parser.add_argument("--length-step", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--aggressive-memory", type=bool, default=True)
    parser.add_argument("--save-dir", type=str, default="results/entropy")
    parser.add_argument("--save-file", type=str, default=None)
    return parser


def generate_save_filename(args) -> str:
    """llama-7b_none.json  /  llama-7b_linear_dynamic.json  / …"""
    model_name = args.model_name.split("/")[-1]
    parts = [model_name, args.rope_type]
    if args.rope_type != "none":
        if args.rope_factor is not None:
            parts.append(f"factor{str(args.rope_factor).replace('.', '_')}")
        elif args.rope_dynamic:
            parts.append("dynamic")
    return "_".join(parts) + ".json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    os.environ.setdefault("USE_FLASH_ATTN", "0")

    parser = argparse.ArgumentParser(description="Attention-entropy evaluation.")
    parser = add_args_model(parser)
    parser = add_args_entropy(parser)
    args = parser.parse_args()

    # experiment-spec defaults
    args.model_name = args.model_name or "huggyllama/llama-7b"
    args.rope_type = "freq-reciprocal"
    args.rope_dynamic = True
    args.min_length = 512
    args.max_length = 3072

    model, _ = load_model(args)
    tokenizer = load_tokenizer(args)
    args.save_file = args.save_file or generate_save_filename(args)

    evaluator = EntropyEvaluator(
        model=model,
        tokenizer=tokenizer,
        dataset=args.dataset_name,
        split=args.split,
        limit=args.limit,
        device=args.device,
        add_start_token=args.add_start_token,
        max_length=args.max_length,
        min_length=args.min_length,
        length_step=args.length_step,
        top_k=args.top_k,
        aggressive_memory=args.aggressive_memory,
        save_dir=args.save_dir,
        save_file=args.save_file,
    )
    evaluator.evaluate()
