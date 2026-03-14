"""
eval/entropy.py
---------------
Attention-entropy evaluator for LLaMA-style models.

Interface mirrors PerplexityEvaluator so the two can be used interchangeably
from test.py or standalone scripts.  All heavy imports follow the project
convention: models.model_loader.{load_model, load_tokenizer, add_args_model}.

Measured quantity
-----------------
For every forward pass the model returns one attention-weight tensor per layer:

    A  ∈ ℝ^{B × H × T × T}   (causal, already softmax-ed)

Shannon entropy at query position t for head h in layer l:

    entropy(l, h, t) = -∑_{k=0}^{t} A[l, h, t, k] · log(A[l, h, t, k] + ε)

The evaluator computes, *per sequence length*:

    mean_entropy_by_layer      shape [num_layers]
        mean over heads, tokens, samples

    mean_entropy_by_head       shape [num_layers, num_heads]
        mean over tokens, samples

    entropy_by_position        shape [num_layers, seq_len]
        mean over heads, samples  (used for position-curve plot)

    entropy_quartiles_by_layer shape [num_layers, 5]  (min, Q1, med, Q3, max)
        distribution across heads × token positions × samples (for box plot)

    top_k_concentration        shape [seq_len]
        fraction of total attention mass in the top-k keys, mean over
        layers, heads, samples  (measures attention sharpness)

JSON schema saved to disk
-------------------------
{
    "lengths": [256, 512, 1024, 2048],
    "model_name": "huggyllama/llama-7b",
    "rope_type": "none",
    "top_k": 10,
    "results": {
        "256": {
            "mean_entropy_by_layer":       [...],          # list[float] len=num_layers
            "mean_entropy_by_head":        [[...]],        # list[list[float]] num_layers×num_heads
            "entropy_by_position":         [[...]],        # list[list[float]] num_layers×seq_len
            "entropy_quartiles_by_layer":  [[...]],        # list[list[float]] num_layers×5
            "top_k_concentration":         [...]           # list[float] len=seq_len
        },
        ...
    }
}
"""

import gc
import math
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import time
import argparse

import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from models.model_loader import load_model, load_tokenizer, add_args_model


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_EPS = 1e-9


def _shannon_entropy(attn: torch.Tensor) -> torch.Tensor:
    """
    Compute Shannon entropy along the last (key) dimension.

    Parameters
    ----------
    attn : Tensor  shape [..., T_q, T_k]   values in [0, 1], sum-to-1 on last dim

    Returns
    -------
    Tensor  shape [..., T_q]   entropy in nats
    """
    return -(attn * (attn + _EPS).log()).sum(dim=-1)


# ---------------------------------------------------------------------------
# EntropyEvaluator
# ---------------------------------------------------------------------------


class EntropyEvaluator:
    r"""
    Attention-entropy evaluator for LLaMA models.

    Attributes
    ----------
    model            : LlamaForCausalLM  (eval mode)
    tokenizer        : PreTrainedTokenizer
    dataset_path     : str
    split            : str
    limit            : int | None
    device           : str
    add_start_token  : bool
    max_length       : int   upper bound of length_list
    min_length       : int   lower bound of length_list (first entry)
    length_step      : int | None
    top_k            : int   for concentration metric
    aggressive_memory: bool
    length_list      : list[int]
    save_dir         : str
    save_file        : str
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
        model            : pretrained LlamaForCausalLM
        tokenizer        : matching tokenizer
        dataset          : HuggingFace dataset path (must have 'input_ids' column)
        split            : dataset split
        limit            : number of samples; None → use all
        device           : 'cuda' / 'cpu' / 'gpu' / None (auto)
        add_start_token  : prepend BOS token (model must have bos_token when True)
        max_length       : maximum sequence length to evaluate
        min_length       : minimum sequence length (first point)
        length_step      : fixed step between lengths; None → exponential ×2
        top_k            : k for top-k concentration metric
        aggressive_memory: free GPU cache after every sample
        save_dir         : directory for JSON output
        save_file        : JSON filename; auto-generated when None
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

        # ── device ────────────────────────────────────────────────────────
        if device is not None:
            assert device in (
                "gpu",
                "cpu",
                "cuda",
            ), "device must be 'gpu', 'cpu', or 'cuda'."
            self.device = "cuda" if device == "gpu" else device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ── BOS sanity check ───────────────────────────────────────────────
        if self.add_start_token:
            assert self.tokenizer.bos_token is not None, (
                "add_start_token=True requires the model/tokenizer to have a BOS "
                "token.  Pass add_start_token=False or use a different model."
            )

        # ── dataset ───────────────────────────────────────────────────────
        self.dataset = load_dataset(self.dataset_path, split=self.split)
        print(f"Dataset size: {len(self.dataset)}")
        if self.limit is not None:
            self.dataset = self.dataset[: self.limit]

        # ── length list ───────────────────────────────────────────────────
        self.length_list = self._generate_length_list()
        print(f"Length list: {self.length_list}")

        # ── output ────────────────────────────────────────────────────────
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_file = (
            save_file if save_file else f"{time.strftime('%Y%m%d-%H%M%S')}.json"
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _generate_length_list(self) -> list:
        """
        Build the list of context lengths to evaluate.

        Exponential (×2) when length_step is None; fixed step otherwise.
        """
        if self.length_step is not None:
            return list(range(self.min_length, self.max_length + 1, self.length_step))

        lst = [self.min_length]
        while True:
            nxt = lst[-1] * 2
            if nxt <= self.max_length:
                lst.append(nxt)
            else:
                break
        return lst

    # ------------------------------------------------------------------
    # core computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_entropy_at_length(self, max_length: int) -> dict:
        """
        Run all samples at *max_length* and return aggregated entropy stats.

        Parameters
        ----------
        max_length : int   tokens per sample (BOS included when add_start_token)

        Returns
        -------
        dict with keys:
            mean_entropy_by_layer      list[float]             len=num_layers
            mean_entropy_by_head       list[list[float]]       num_layers × num_heads
            entropy_by_position        list[list[float]]       num_layers × max_length
            entropy_quartiles_by_layer list[list[float]]       num_layers × 5
            top_k_concentration        list[float]             len=max_length
        """
        encoded_texts = self.dataset["input_ids"]
        attn_masks = self.dataset["attention_mask"]

        max_tokenized_len = (max_length - 1) if self.add_start_token else max_length

        # truncate every sample to max_tokenized_len
        encoded_texts = [x[:max_tokenized_len] for x in encoded_texts]
        attn_masks = [x[:max_tokenized_len] for x in attn_masks]

        # accumulators — populated after the first sample when we know num_layers / num_heads
        acc_entropy = None  # shape [num_layers, num_heads, max_length]
        acc_topk = None  # shape [max_length]
        n_samples = 0

        # per-sample flat collections for quartile computation (layer-wise)
        # head_token_entropy[l] = list of scalar entropy values
        head_token_entropy: list[list] = []

        pbar = tqdm(
            total=len(encoded_texts),
            desc=f"Entropy @ len={max_length}",
            leave=False,
        )

        for idx in range(len(encoded_texts)):
            # ── build input ────────────────────────────────────────────
            labels = torch.tensor(encoded_texts[idx : idx + 1])  # [1, T]
            attn_mask = torch.tensor(attn_masks[idx : idx + 1])  # [1, T]

            if self.add_start_token:
                bos = torch.tensor([[self.tokenizer.bos_token_id]])
                labels = torch.cat([bos, labels], dim=1)
                bos_mask = torch.ones((1, 1), dtype=attn_mask.dtype)
                attn_mask = torch.cat([bos_mask, attn_mask], dim=1)

            input_ids = labels.to(self.device)
            attn_mask = attn_mask.to(self.device)
            seq_len = input_ids.size(1)

            # ── forward pass ────────────────────────────────────────────
            outputs = self.model(
                input_ids,
                attention_mask=attn_mask,
                output_attentions=True,
            )
            # outputs.attentions: tuple of num_layers tensors [1, H, T, T]

            attentions = outputs.attentions  # tuple len=num_layers
            num_layers = len(attentions)
            num_heads = attentions[0].size(1)

            # lazy initialisation
            if acc_entropy is None:
                acc_entropy = torch.zeros(
                    num_layers,
                    num_heads,
                    max_length,
                    dtype=torch.float64,
                )
                acc_topk = torch.zeros(max_length, dtype=torch.float64)
                head_token_entropy = [[] for _ in range(num_layers)]

            # ── entropy per (layer, head, position) ─────────────────────
            # attn[l]: [1, H, T, T]  → entropy: [H, T]
            for l_idx, attn in enumerate(attentions):
                # cast to float32 for numerical stability even with 4-bit model
                a = attn[0].float()  # [H, T_q, T_k]
                ent = _shannon_entropy(a)  # [H, T_q]
                # pad to max_length if seq_len < max_length
                if seq_len < max_length:
                    pad = torch.zeros(
                        num_heads,
                        max_length - seq_len,
                        dtype=ent.dtype,
                        device=ent.device,
                    )
                    ent = torch.cat([ent, pad], dim=1)  # [H, max_length]
                acc_entropy[l_idx] += ent.cpu().double()
                # collect all (head × token) entropy values for quartile
                head_token_entropy[l_idx].extend(
                    ent[:, :seq_len].cpu().numpy().ravel().tolist()
                )

            # ── top-k concentration ──────────────────────────────────────
            # use layer 0 head 0 as representative (averaged below)
            # aggregate over all layers & heads for robustness
            topk_vals = torch.zeros(max_length, dtype=torch.float64)
            for attn in attentions:
                a = attn[0].float()  # [H, T_q, T_k]
                # top-k mass at each query position
                k = min(self.top_k, a.size(-1))
                tkv = a.topk(k, dim=-1).values.sum(-1)  # [H, T_q]
                tkv_mean = tkv.mean(dim=0)  # [T_q]
                if seq_len < max_length:
                    pad = torch.zeros(
                        max_length - seq_len,
                        dtype=tkv_mean.dtype,
                        device=tkv_mean.device,
                    )
                    tkv_mean = torch.cat([tkv_mean, pad])
                topk_vals += tkv_mean.cpu().double()
            topk_vals /= num_layers
            acc_topk += topk_vals

            n_samples += 1

            # ── memory ──────────────────────────────────────────────────
            del outputs, attentions, input_ids, attn_mask
            if self.aggressive_memory:
                gc.collect()
                torch.cuda.empty_cache()

            pbar.update(1)

        pbar.close()

        if n_samples == 0:
            raise RuntimeError("No samples were processed.")

        # ── aggregate ───────────────────────────────────────────────────
        avg_entropy = acc_entropy / n_samples  # [num_layers, num_heads, max_length]
        avg_topk = (acc_topk / n_samples).tolist()  # [max_length]

        # mean_entropy_by_layer: mean over heads and token positions
        mean_by_layer = avg_entropy.mean(dim=(1, 2)).tolist()  # [num_layers]

        # mean_entropy_by_head: mean over token positions
        mean_by_head = avg_entropy.mean(dim=2).tolist()  # [num_layers, num_heads]

        # entropy_by_position: mean over heads
        by_position = avg_entropy.mean(dim=1).tolist()  # [num_layers, max_length]

        # quartiles per layer from raw head × token distributions
        quartiles = []
        for l_idx in range(num_layers):
            vals = np.array(head_token_entropy[l_idx], dtype=np.float64)
            q = np.quantile(vals, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist()
            quartiles.append(q)  # [5]

        return {
            "mean_entropy_by_layer": mean_by_layer,
            "mean_entropy_by_head": mean_by_head,
            "entropy_by_position": by_position,
            "entropy_quartiles_by_layer": quartiles,
            "top_k_concentration": avg_topk,
        }

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def evaluate(self) -> dict:
        r"""
        Evaluate attention entropy at all lengths in ``self.length_list``.

        Returns
        -------
        dict  matching the JSON schema described in the module docstring.
        """
        results = {}

        outer = tqdm(self.length_list, desc="Entropy Evaluation")
        for seq_len in outer:
            outer.set_postfix(seq_len=seq_len)
            torch.cuda.empty_cache()
            results[str(seq_len)] = self._compute_entropy_at_length(seq_len)

        output = {
            "lengths": self.length_list,
            "model_name": getattr(self.model.config, "_name_or_path", "unknown"),
            "top_k": self.top_k,
            "results": results,
        }

        save_path = os.path.join(self.save_dir, self.save_file)
        with open(save_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved → {save_path}")

        return output


# ---------------------------------------------------------------------------
# argument helpers
# ---------------------------------------------------------------------------


def add_args_entropy(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Register entropy-evaluation arguments with *parser*.

    Follows the same pattern as add_args_perplexity so both can be combined
    in test.py or called independently.
    """
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="emozilla/proofpile-test-tokenized",
        help="HuggingFace dataset (must have 'input_ids' column).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of samples to evaluate (None → all).",
    )
    parser.add_argument(
        "--add-start-token",
        type=bool,
        default=True,
        help="Prepend BOS token to each sample.",
    )
    parser.add_argument(
        "--length-step",
        type=int,
        default=None,
        help="Fixed step between evaluated lengths; None → exponential ×2.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="k for top-k attention-concentration metric.",
    )
    parser.add_argument(
        "--aggressive-memory",
        type=bool,
        default=True,
        help="Free GPU cache after every sample.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/entropy",
        help="Directory to write JSON results.",
    )
    parser.add_argument(
        "--save-file",
        type=str,
        default=None,
        help="JSON filename (auto-generated when omitted).",
    )
    return parser


def generate_save_filename(args) -> str:
    """
    Build a deterministic filename from model + RoPE config.

    Examples
    --------
    --rope-type none                              → llama-7b_none.json
    --rope-type linear --rope-dynamic             → llama-7b_linear_dynamic.json
    --rope-type linear --rope-factor 4.0          → llama-7b_linear_factor4_0.json
    """
    model_name = args.model_name.split("/")[-1]
    parts = [model_name, args.rope_type]

    if args.rope_type != "none":
        if args.rope_factor is not None:
            factor_str = str(args.rope_factor).replace(".", "_")
            parts.append(f"factor{factor_str}")
        elif args.rope_dynamic:
            parts.append("dynamic")

    return "_".join(parts) + ".json"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    os.environ.setdefault("USE_FLASH_ATTN", "0")  # flash-attn hides attn weights

    parser = argparse.ArgumentParser(
        description="Attention-entropy evaluation for LLaMA models."
    )
    parser = add_args_model(parser)
    parser = add_args_entropy(parser)
    args = parser.parse_args()

    # Fixed per experiment spec:
    #   model  = huggyllama/llama-7b
    #   rope   = none
    #   lengths= 256 / 512 / 1024 / 2048
    args.model_name = args.model_name or "huggyllama/llama-7b"
    args.rope_type = "none"
    args.min_length = 256
    args.max_length = 2048

    model, config = load_model(args)
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

    results = evaluator.evaluate()
    for length in results["lengths"]:
        r = results["results"][str(length)]
        me = sum(r["mean_entropy_by_layer"]) / len(r["mean_entropy_by_layer"])
        print(f"  seq_len={length:>5d}  global_mean_entropy={me:.4f} nats")
