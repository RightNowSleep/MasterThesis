#!/usr/bin/env python3
"""Optuna-based hyperparameter search for inverse-dual-rope-scaled attention scaling.

Searches optimal (alpha, beta, gamma) parameters for the decomposed scaling function:
    s(t) = (1 + α·ln(k+1)) · (1 + β·e^(-γr))

Uses Optuna TPE sampler with MedianPruner for efficient Bayesian optimization.

Usage Examples:
    # Traditional: Search on base model
    python search_attn_scale_params.py --model-name huggyllama/llama-7b --rope-type inverse-dual-rope-scaled

    # New: Search on top of a trained inverse-dual-rope adapter
    python search_attn_scale_params.py \\
        --model-name huggyllama/llama-7b \\
        --base-adapter-path finetunes/continued_pretrain/inverse-dual-rope_20260403_103555 \\
        --rope-type inverse-dual-rope-scaled \\
        --rope-dynamic
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import torch
import gc

from models.model_loader import load_model, load_tokenizer, add_args_model
from eval.perplexity import PerplexityEvaluator


def parse_args():
    """Parse CLI arguments.

    Model-related arguments (including --base-adapter-path) are automatically
    included via add_args_model() from models.model_loader.
    """
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for inverse-dual-rope-scaled"
    )

    # ── Model arguments (reuse from model_loader) ──────────────────────
    parser = add_args_model(parser)

    # ── Optuna arguments ───────────────────────────────────────────────
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials (default: 100)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="inverse-dual-rope-scaled-search",
        help="Optuna study name",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///results/param_search/optuna.db",
        help="Optuna storage URL. Default: SQLite database for dashboard visualization.",
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=42,
        help="Random seed for TPE sampler (default: 42)",
    )
    parser.add_argument(
        "--pruner-n-warmup-steps",
        type=int,
        default=10,
        help="Number of warmup steps before MedianPruning starts (default: 10)",
    )
    parser.add_argument(
        "--pruner-n-min-steps",
        type=int,
        default=5,
        help="Minimum trials in each comparison group for MedianPruner (default: 5)",
    )

    # ── Search space bounds ────────────────────────────────────────────
    parser.add_argument(
        "--alpha-range",
        type=str,
        default="0.05,0.40",
        help="Alpha search range as 'min,max' (default: '0.05,0.40')",
    )
    parser.add_argument(
        "--beta-range",
        type=str,
        default="0.20,1.50",
        help="Beta search range as 'min,max' (default: '0.20,1.50')",
    )
    parser.add_argument(
        "--gamma-range",
        type=str,
        default="0.50,5.00",
        help="Gamma search range as 'min,max' (default: '0.50,5.00')",
    )

    # ── Evaluation arguments ───────────────────────────────────────────
    parser.add_argument("--eval-min-length", type=int, default=4096)
    parser.add_argument("--eval-max-length", type=int, default=65536)
    parser.add_argument(
        "--eval-dataset",
        type=str,
        default="emozilla/proofpile-test-tokenized",
    )
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--eval-limit", type=int, default=50)

    # ── Output ─────────────────────────────────────────────────────────
    parser.add_argument("--output-dir", type=str, default="results/param_search")

    return parser.parse_args()


def set_inverse_dual_rope_scaled_params(model, alpha: float, beta: float, gamma: float):
    """Set alpha/beta/gamma parameters on all LlamaInverseDualRoPEScaledEmbedding layers.

    Iterates over all named modules and sets attributes on any module that has
    'alpha', 'beta', or 'gamma' attributes. After updating parameters, invalidates
    all cos/sin caches so that the next forward pass recomputes embeddings with
    the new parameter values.

    Note:
        When --base-adapter-path is used, this function modifies scaling parameters
        on top of an already-loaded base adapter, allowing incremental fine-tuning
        of RoPE scaling behavior.
    """
    for name, module in model.named_modules():
        if hasattr(module, "alpha"):
            module.alpha = alpha
        if hasattr(module, "beta"):
            module.beta = beta
        if hasattr(module, "gamma"):
            module.gamma = gamma
        if hasattr(module, "max_seq_len_cached"):
            module.max_seq_len_cached = 0
        if hasattr(module, "_dynamic_seq_len_cached"):
            module._dynamic_seq_len_cached = -1
            module._dynamic_s_cached = -1.0


def objective(trial, model, tokenizer, args):
    """Optuna objective function: evaluate (alpha, beta, gamma) and return weighted ppl."""
    # Suggest parameters
    alpha_min, alpha_max = map(float, args.alpha_range.split(","))
    beta_min, beta_max = map(float, args.beta_range.split(","))
    gamma_min, gamma_max = map(float, args.gamma_range.split(","))

    alpha = trial.suggest_float("alpha", alpha_min, alpha_max, log=True)
    beta = trial.suggest_float("beta", beta_min, beta_max)
    gamma = trial.suggest_float("gamma", gamma_min, gamma_max, log=True)

    print(
        f"\n[Trial {trial.number}] alpha={alpha:.4f}, beta={beta:.4f}, gamma={gamma:.4f}"
    )

    # Set parameters
    set_inverse_dual_rope_scaled_params(model, alpha, beta, gamma)

    # Evaluate
    evaluator = PerplexityEvaluator(
        model=model,
        tokenizer=tokenizer,
        dataset=args.eval_dataset,
        split=args.eval_split,
        limit=args.eval_limit,
        device=args.device or "gpu",
        add_start_token=True,
        max_length=args.eval_max_length,
        min_length=args.eval_min_length,
        length_step=None,
        sliding_window=256,
        truncate=True,
        aggressive_memory=True,
        save_dir="/tmp",
        save_file=f"optuna_trial_{trial.number}_{time.time()}.json",
    )

    results = evaluator.evaluate()
    lengths = results["lengths"]
    perplexities = results["perplexities"]

    torch.cuda.empty_cache()
    gc.collect()

    weights = [length / sum(lengths) for length in lengths]
    weighted_ppl = sum(p * w for p, w in zip(perplexities, weights))

    print(f"  [Result] weighted_ppl={weighted_ppl:.4f}, lengths={lengths}")
    print(f"  [Result] perplexities={[f'{p:.4f}' for p in perplexities]}")

    # Report intermediate value for pruning
    trial.report(weighted_ppl, step=0)

    if trial.should_prune():
        raise optuna.TrialPruned()

    return weighted_ppl


def save_results(study, output_dir: str, args):
    """Save Optuna study results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optuna_search_results_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    best_trial = study.best_trial

    output_data = {
        "args": vars(args),
        "best_params": best_trial.params,
        "best_value": best_trial.value,
        "n_trials": len(study.trials),
        "n_successful": len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        ),
        "n_pruned": len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        ),
        "all_trials": [
            {
                "number": t.number,
                "params": t.params,
                "value": t.value,
                "state": str(t.state),
            }
            for t in study.trials
        ],
        "timestamp": timestamp,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {filepath}")
    print(f"Best parameters:")
    for key, val in best_trial.params.items():
        print(f"  {key}: {val:.6f}")
    print(f"Best weighted perplexity: {best_trial.value:.4f}")


def main():
    """Entry point."""
    args = parse_args()

    print("=" * 60)
    print("Optuna Hyperparameter Search for inverse-dual-rope-scaled")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"RoPE type: {args.rope_type}")
    print(f"Trials: {args.n_trials}")
    alpha_min, alpha_max = map(float, args.alpha_range.split(","))
    beta_min, beta_max = map(float, args.beta_range.split(","))
    gamma_min, gamma_max = map(float, args.gamma_range.split(","))
    print(
        f"Search space: alpha=[{alpha_min},{alpha_max}], beta=[{beta_min},{beta_max}], gamma=[{gamma_min},{gamma_max}]"
    )
    print(
        f"Evaluation: [{args.eval_min_length}, {args.eval_max_length}] on {args.eval_dataset}"
    )
    print(f"Output: {args.output_dir}")
    if args.base_adapter_path:
        print(f"Base adapter : {args.base_adapter_path}")
        print("(Loading base RoPE method adapter before applying target scaling)")
    print("=" * 60)

    if args.base_adapter_path:
        print(f"[INFO] Will load base adapter from: {args.base_adapter_path}")
        print(f"[INFO] Then apply RoPE type: {args.rope_type} with dynamic scaling")

    print("\nLoading model...")
    model, config = load_model(args)
    tokenizer = load_tokenizer(args)
    print(f"Model loaded | device: {next(model.parameters()).device}")

    # Create Optuna study
    sampler = TPESampler(seed=args.sampler_seed)
    pruner = MedianPruner(
        n_startup_trials=args.pruner_n_warmup_steps,
        n_warmup_steps=args.pruner_n_warmup_steps,
        interval_steps=1,
        n_min_trials=args.pruner_n_min_steps,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=True,
    )

    print(f"\nStudy created: '{study.study_name}'")
    print(f"Sampler: TPE (seed={args.sampler_seed})")
    print(
        f"Pruner: MedianPruner(warmup={args.pruner_n_warmup_steps}, min={args.pruner_n_min_steps})"
    )
    print(f"Storage: {args.storage or '(in-memory)'}")
    print(f"\nStarting optimization ({args.n_trials} trials)...")
    print("-" * 60)

    start_time = time.time()

    study.optimize(
        lambda trial: objective(trial, model, tokenizer, args),
        n_trials=args.n_trials,
        show_progress_bar=False,
    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("Optimization Complete!")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f}s")
    print(
        f"Trials completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}"
    )
    print(
        f"Trials pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}"
    )
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best params: {study.best_params}")
    print(f"Best weighted perplexity: {study.best_value:.4f}")

    save_results(study, args.output_dir, args)

    print("\n" + "=" * 60)
    print("Search completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
