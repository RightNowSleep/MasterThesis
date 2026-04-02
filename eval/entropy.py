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

Stored metrics (evaluated at a single max_length)
--------------------------------------------------
  entropy_head_layer_position      [L][H][T]   raw,  mean over samples only  ← primary
  norm_entropy_head_layer_position [L][H][T]   norm, mean over samples only  ← primary
  entropy_layer_position           [L][T]      raw,  mean over H (derived)
  norm_entropy_layer_position      [L][T]      norm, mean over H (derived)
  entropy_head_layer               [L][H]      raw,  mean over T (derived)
  norm_entropy_head_layer          [L][H]      norm, mean over T (derived)
  head_norm_std_by_layer           [L]         std of per-head H_norm across H
  raw_entropy_quartiles_by_layer   [L][5]      (min,Q1,med,Q3,max) of raw H
  norm_entropy_quartiles_by_layer  [L][5]      same for normalised H
  top_k_concentration              [T]         top-k attn mass fraction
  top_k_boundary                   int         first position with window > k
"""

import gc
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import random
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

    Args:
        attn: Tensor of shape [..., T_q, T_k] with softmax weights, rows sum to 1.

    Returns:
        Tensor of shape [..., T_q].
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

    Args:
        ent: Tensor of shape [..., T].

    Returns:
        Tensor of shape [..., T] with values in [0, 1].
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

    Attributes:
        model: LlamaForCausalLM in eval mode.
        tokenizer: PreTrainedTokenizer.
        dataset_path: Dataset path or name.
        split: Dataset split.
        num_samples: Number of randomly selected samples (all length ≥ max_length).
        device: Compute device.
        add_start_token: Whether to prepend BOS token.
        max_length: Single evaluation length.
        top_k: K for top-k concentration metric.
        aggressive_memory: Whether to free GPU cache after every sample.
        seed: Random seed for reproducible sample selection.
        save_dir: Directory for JSON output.
        save_file: JSON filename (auto-generated when None).
    """

    def __init__(
        self,
        model,
        tokenizer,
        dataset: str = "emozilla/proofpile-test-tokenized",
        split: str = "test",
        num_samples: int = 20,
        device: str = None,
        add_start_token: bool = True,
        max_length: int = 2048,
        top_k: int = 10,
        aggressive_memory: bool = True,
        seed: int = 42,
        save_dir: str = "results/entropy",
        save_file: str = None,
    ):
        r"""
        Initialize the entropy evaluator.

        Args:
            model: Pretrained LlamaForCausalLM.
            tokenizer: Matching tokenizer.
            dataset: HuggingFace dataset path (needs 'input_ids' column).
            split: Dataset split.
            num_samples: How many samples to randomly draw (all must be ≥ max_length).
            device: 'cuda' | 'cpu' | 'gpu' | None (auto).
            add_start_token: Prepend BOS; tokenizer must have bos_token.
            max_length: The single sequence length to evaluate.
            top_k: K for the top-k concentration metric.
            aggressive_memory: Free GPU cache after every sample.
            seed: Random seed for reproducible sample selection.
            save_dir: Directory for JSON output.
            save_file: JSON filename (auto-generated when None).
        """
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.dataset_path = dataset
        self.split = split
        self.num_samples = num_samples
        self.add_start_token = add_start_token
        self.max_length = max_length
        self.top_k = top_k
        self.aggressive_memory = aggressive_memory
        self.seed = seed

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

        # ── dataset loading & filtering ───────────────────────────────────
        # We need each raw sample to have at least max_tok tokens so that
        # after optional BOS prepending the sequence is exactly max_length.
        max_tok = (max_length - 1) if self.add_start_token else max_length

        raw = load_dataset(self.dataset_path, split=self.split)
        print(f"Dataset size (total)       : {len(raw)}")

        # Keep only samples with sufficient length
        valid_indices = [
            i for i, ids in enumerate(raw["input_ids"]) if len(ids) >= max_tok
        ]
        print(f"Samples with length ≥ {max_tok}: {len(valid_indices)}")

        if len(valid_indices) < num_samples:
            raise ValueError(
                f"Only {len(valid_indices)} samples have length ≥ {max_tok}, "
                f"but num_samples={num_samples} were requested."
            )

        # Randomly select num_samples indices (reproducible)
        rng = random.Random(self.seed)
        chosen = sorted(rng.sample(valid_indices, num_samples))

        # Truncate to exactly max_tok tokens
        self.encoded = [raw["input_ids"][i][:max_tok] for i in chosen]
        self.masks = [raw["attention_mask"][i][:max_tok] for i in chosen]
        print(f"Randomly selected samples  : {len(self.encoded)} (seed={seed})")

        # output
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_file = save_file or f"{time.strftime('%Y%m%d-%H%M%S')}.json"

    # ------------------------------------------------------------------
    # core computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_entropy(self) -> dict:
        """
        Accumulate attention-entropy statistics over all selected samples.

        All samples are guaranteed to have length == max_length, so no
        padding is ever needed and the accumulator shape is exact from the
        start.

        Returns:
            Dictionary containing entropy metrics with primary output shape
            [L, H, max_length] (mean over samples only).
        """
        max_length = self.max_length

        # Running accumulators – lazily initialised after the first forward pass
        acc_raw = None  # [L, H, max_length]  float64
        acc_norm = None  # [L, H, max_length]  float64
        acc_topk = None  # [max_length]         float64
        n = 0

        pbar = tqdm(
            total=len(self.encoded),
            desc=f"  seq_len={max_length}",
            leave=False,
        )

        for idx in range(len(self.encoded)):
            # ── construct input tensors ───────────────────────────────
            lbl = torch.tensor(self.encoded[idx : idx + 1])  # [1, T]
            mask = torch.tensor(self.masks[idx : idx + 1])  # [1, T]

            if self.add_start_token:
                bos = torch.tensor([[self.tokenizer.bos_token_id]])
                lbl = torch.cat([bos, lbl], dim=1)
                mask = torch.cat([torch.ones((1, 1), dtype=mask.dtype), mask], dim=1)

            ids = lbl.to(self.device)
            mask = mask.to(self.device)
            # T == max_length is guaranteed by construction

            # ── forward pass ─────────────────────────────────────────
            outputs = self.model(ids, attention_mask=mask, output_attentions=True)
            attentions = outputs.attentions  # tuple len=L, each [1, H, T, T]
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

                # No padding needed: every sample has exactly max_length tokens
                acc_raw[l_idx] += ent
                acc_norm[l_idx] += nrm

                # top-k concentration: mean over heads → [T]
                k = min(self.top_k, a.size(-1))
                tkv = a.topk(k, dim=-1).values.sum(-1).mean(0)  # [T]
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
        # Primary tensors: [L, H, max_length] – mean over samples only
        avg_raw = (acc_raw / n).numpy()  # [L, H, T]
        avg_norm = (acc_norm / n).numpy()  # [L, H, T]
        avg_topk = (acc_topk / n).numpy()  # [T]

        # ------------------------------------------------------------------
        # Compute and package all output metrics
        # ------------------------------------------------------------------

        # Primary: [L, H, T]  – mean over samples only (no further reduction)
        ehlp = np.round(avg_raw, _PREC).tolist()
        nehlp = np.round(avg_norm, _PREC).tolist()

        # Derived: [L, T]  – additionally mean over H
        elp = np.round(avg_raw.mean(axis=1), _PREC).tolist()
        nelp = np.round(avg_norm.mean(axis=1), _PREC).tolist()

        # Derived: [L, H]  – additionally mean over T
        ehl = np.round(avg_raw.mean(axis=2), _PREC).tolist()
        nehl = np.round(avg_norm.mean(axis=2), _PREC).tolist()

        # [L]  – intra-layer head specialisation (std of per-head H_norm mean over T)
        hstd = np.round(avg_norm.mean(axis=2).std(axis=1), _PREC).tolist()

        # [L, 5]  – quartile distributions over H×T
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
            # ── primary outputs ─────────────────────────────────────
            "entropy_head_layer_position": ehlp,  # [L][H][T]
            "norm_entropy_head_layer_position": nehlp,  # [L][H][T]
            # ── derived outputs ─────────────────────────────────────
            "entropy_layer_position": elp,  # [L][T]
            "norm_entropy_layer_position": nelp,  # [L][T]
            "entropy_head_layer": ehl,  # [L][H]
            "norm_entropy_head_layer": nehl,  # [L][H]
            "head_norm_std_by_layer": hstd,  # [L]
            "raw_entropy_quartiles_by_layer": raw_q,  # [L][5]
            "norm_entropy_quartiles_by_layer": nrm_q,  # [L][5]
            "top_k_concentration": np.round(avg_topk, _PREC).tolist(),  # [T]
            "top_k_boundary": int(min(self.top_k, max_length)) - 1,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self) -> dict:
        r"""
        Evaluate at ``self.max_length`` and write JSON.

        Returns:
            Full output dictionary including metadata and results.
        """
        torch.cuda.empty_cache()
        results = self._compute_entropy()

        num_layers = len(results["entropy_layer_position"])
        num_heads = len(results["entropy_head_layer"][0])

        output = {
            "max_length": self.max_length,
            "num_samples": self.num_samples,
            "seed": self.seed,
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
        mh = float(np.mean(results["entropy_layer_position"]))
        mn = float(np.mean(results["norm_entropy_layer_position"]))
        print(
            f"\n  seq_len={self.max_length}  mean H={mh:.4f} nats  mean H_norm={mn:.4f}"
        )

        return output


# ---------------------------------------------------------------------------
# Argument helpers  (mirror PerplexityEvaluator convention)
# ---------------------------------------------------------------------------


def add_args_entropy(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Register entropy-evaluation CLI arguments.

    Args:
        parser: ArgumentParser to add arguments to.

    Returns:
        Parser with added arguments.
    """
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="emozilla/proofpile-test-tokenized",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--add-start-token", type=bool, default=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--aggressive-memory", type=bool, default=True)
    parser.add_argument("--save-dir", type=str, default="results/entropy")
    parser.add_argument("--save-file", type=str, default=None)
    return parser


def generate_save_filename(args) -> str:
    """
    Generate filename based on model, adapter, and RoPE configuration.

    Args:
        args: Arguments object with model_name, adapter_path (optional), rope_type,
              rope_factor, and rope_dynamic.

    Returns:
        Filename string. Examples:
            - Adapter-only: 'llama-7b_adapter_dual-rope_20260402_113443.json'
            - RoPE-only: 'llama-7b_linear_dynamic.json'
            - Adapter+RoPE: 'llama-7b_adapter_dual-rope_20260402_113443_linear_dynamic.json'
            - No adapter or RoPE: 'llama-7b_none.json'
    """
    model_name = args.model_name.split("/")[-1]
    parts = [model_name]

    if hasattr(args, "adapter_path") and args.adapter_path:
        adapter_id = os.path.basename(args.adapter_path.rstrip("/"))
        parts.append("adapter")
        parts.append(adapter_id)

    rope_type = getattr(args, "rope_type", "none")
    if rope_type != "none":
        parts.append(rope_type)
        rope_factor = getattr(args, "rope_factor", None)
        rope_dynamic = getattr(args, "rope_dynamic", False)

        if rope_factor is not None:
            parts.append(f"factor{str(rope_factor).replace('.', '_')}")
        elif rope_dynamic:
            parts.append("dynamic")
    else:
        if not (hasattr(args, "adapter_path") and args.adapter_path):
            parts.append("none")

    return "_".join(parts) + ".json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.environ.setdefault("USE_FLASH_ATTN", "0")

    parser = argparse.ArgumentParser(description="Attention-entropy evaluation.")
    parser = add_args_model(parser)
    parser = add_args_entropy(parser)
    args = parser.parse_args()

    # experiment-spec defaults
    args.model_name = args.model_name or "huggyllama/llama-7b"

    model, _ = load_model(args)
    tokenizer = load_tokenizer(args)
    args.save_file = args.save_file or generate_save_filename(args)

    evaluator = EntropyEvaluator(
        model=model,
        tokenizer=tokenizer,
        dataset=args.dataset_name,
        split=args.split,
        num_samples=args.num_samples,
        device=args.device,
        add_start_token=args.add_start_token,
        max_length=args.max_length,
        top_k=args.top_k,
        aggressive_memory=args.aggressive_memory,
        seed=args.seed,
        save_dir=args.save_dir,
        save_file=args.save_file,
    )
    evaluator.evaluate()
