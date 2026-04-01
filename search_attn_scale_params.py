#!/usr/bin/env python3
"""
Attention Scale Parameter Search Script

This script searches for optimal parameters in the attention scaling formula:
    t_base = attn_scale_base + attn_scale_coef * torch.log(s)

where attn_scale_base and attn_scale_coef are the parameters to optimize.
The optimization objective is to minimize perplexity at 64K context length.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple, Optional

import torch
import gc
import math
import numpy as np

from models.model_loader import load_model, load_tokenizer, add_args_model
from eval.perplexity import PerplexityEvaluator

_eval_cache = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search for optimal attention scale parameters"
    )

    parser = add_args_model(parser)

    parser.add_argument(
        "--search-method",
        type=str,
        default="adaptive",
        choices=["grid", "random", "bayesian", "adaptive", "log-scale", "bohb"],
        help="Search method: grid, random, bayesian, adaptive (multi-stage), log-scale, or bohb",
    )

    parser.add_argument(
        "--attn-scale-coef-min",
        type=float,
        default=0.010,
        help="Minimum value for attn_scale_coef parameter (default: 0.010, 3 decimal places)",
    )

    parser.add_argument(
        "--attn-scale-coef-max",
        type=float,
        default=0.300,
        help="Maximum value for attn_scale_coef parameter (default: 0.300, 3 decimal places)",
    )

    parser.add_argument(
        "--attn-scale-coef-steps",
        type=int,
        default=100,
        help="Number of steps for attn_scale_coef in grid search (default: 100, dense sampling)",
    )

    parser.add_argument(
        "--random-samples",
        type=int,
        default=50,
        help="Number of random samples for random search",
    )

    parser.add_argument(
        "--bayesian-iterations",
        type=int,
        default=100,
        help="Number of iterations for bayesian optimization",
    )

    parser.add_argument(
        "--bohb-initial-samples",
        type=int,
        default=25,
        help="Number of initial random samples for BOHB",
    )

    parser.add_argument(
        "--bohb-iterations",
        type=int,
        default=100,
        help="Number of iterations for BOHB optimization",
    )

    parser.add_argument(
        "--bohb-early-stop-factor",
        type=int,
        default=3,
        help="Early stop factor for BOHB low-fidelity evaluation (eval_limit // factor)",
    )

    parser.add_argument(
        "--adaptive-stages",
        type=int,
        default=6,
        help="Number of stages for adaptive search (coarse to fine)",
    )

    parser.add_argument(
        "--adaptive-refinement-factor",
        type=float,
        default=0.3,
        help="Factor to narrow search space in each adaptive stage (0.3 means 30%% of previous range)",
    )

    parser.add_argument(
        "--eval-min-length",
        type=int,
        default=4096,
        help="Minimum context length for perplexity evaluation (default: 4096)",
    )

    parser.add_argument(
        "--eval-max-length",
        type=int,
        default=65536,
        help="Maximum context length for perplexity evaluation (default: 65536)",
    )

    parser.add_argument(
        "--eval-dataset",
        type=str,
        default="emozilla/proofpile-test-tokenized",
        help="Dataset for perplexity evaluation",
    )

    parser.add_argument(
        "--eval-split",
        type=str,
        default="test",
        help="Dataset split for evaluation",
    )

    parser.add_argument(
        "--eval-limit",
        type=int,
        default=50,
        help="Number of samples to evaluate",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/param_search",
        help="Directory to save search results",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from previous search results file",
    )

    return parser.parse_args()


def generate_grid_search_space(
    coef_min: float,
    coef_max: float,
    coef_steps: int,
) -> List[float]:
    """
    Generate parameter values for grid search.

    Returns:
        List of attn_scale_coef values (rounded to 3 decimal places)
    """
    coef_values = torch.linspace(coef_min, coef_max, coef_steps).tolist()
    coef_values = [round(v, 3) for v in coef_values]

    return coef_values


def generate_random_search_space(
    coef_min: float,
    coef_max: float,
    n_samples: int,
) -> List[float]:
    """
    Generate random parameter values.

    Returns:
        List of attn_scale_coef values (rounded to 3 decimal places)
    """
    coef_values = torch.rand(n_samples) * (coef_max - coef_min) + coef_min
    coef_values = [round(v, 3) for v in coef_values.tolist()]

    return coef_values


def set_model_attn_scale_params(model, attn_scale_base: float, attn_scale_coef: float):
    """
    Set attention scale parameters in all rotary embedding layers.

    Args:
        model: The language model
        attn_scale_base: Base value for attention scaling
        attn_scale_coef: Coefficient for attention scaling
    """
    for name, module in model.named_modules():
        if hasattr(module, "attn_scale_base"):
            module.attn_scale_base = attn_scale_base
        if hasattr(module, "attn_scale_coef"):
            module.attn_scale_coef = attn_scale_coef


def evaluate_params(
    model,
    tokenizer,
    attn_scale_coef: float,
    eval_min_length: int,
    eval_max_length: int,
    dataset: str,
    split: str,
    limit: int,
    device: str,
) -> Dict:
    """
    Evaluate a parameter combination using perplexity at multiple context lengths.

    Args:
        model: The language model
        tokenizer: The tokenizer
        attn_scale_coef: Coefficient for attention scaling
        eval_min_length: Minimum context length for evaluation
        eval_max_length: Maximum context length for evaluation
        dataset: Dataset name
        split: Dataset split
        limit: Number of samples
        device: Device to use

    Returns:
        Dictionary containing:
            - weighted_ppl: Weighted average perplexity (lower is better)
            - lengths: List of evaluated context lengths
            - perplexities: List of perplexities at each length
        Uses length-weighted average: longer contexts get higher weights
    """
    global _eval_cache

    cache_key = (attn_scale_coef, eval_min_length, eval_max_length, limit)
    if cache_key in _eval_cache:
        print(f"  [Cache Hit] Returning cached result for coef={attn_scale_coef:.3f}")
        return _eval_cache[cache_key]

    print(f"  [Evaluating] attn_scale_coef={attn_scale_coef:.3f}")
    print(
        f"  [Config] length_range=[{eval_min_length}, {eval_max_length}], samples={limit}"
    )

    attn_scale_base = 1.0
    set_model_attn_scale_params(model, attn_scale_base, attn_scale_coef)

    evaluator = PerplexityEvaluator(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        split=split,
        limit=limit,
        device=device,
        add_start_token=True,
        max_length=eval_max_length,
        min_length=eval_min_length,
        length_step=None,
        sliding_window=256,
        truncate=True,
        aggressive_memory=True,
        save_dir="/tmp",
        save_file=f"temp_{time.time()}.json",
    )

    results = evaluator.evaluate()
    lengths = results["lengths"]
    perplexities = results["perplexities"]

    torch.cuda.empty_cache()
    gc.collect()

    weights = [length / sum(lengths) for length in lengths]
    weighted_ppl = sum(p * w for p, w in zip(perplexities, weights))

    print(f"  [Results] Context lengths: {lengths}")
    print(f"  [Results] Perplexities: {[f'{p:.4f}' for p in perplexities]}")
    print(f"  [Results] Weighted average perplexity: {weighted_ppl:.4f}")

    result = {
        "weighted_ppl": weighted_ppl,
        "lengths": lengths,
        "perplexities": perplexities,
    }

    _eval_cache[cache_key] = result

    return result


def save_results(
    results: List[Dict],
    output_dir: str,
    args,
):
    """
    Save search results to JSON file.

    Args:
        results: List of evaluation results
        output_dir: Output directory
        args: Command line arguments
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"search_results_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    output_data = {
        "args": vars(args),
        "results": results,
        "best_result": min(results, key=lambda x: x["perplexity"]),
        "timestamp": timestamp,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {filepath}")
    print(f"Best parameters:")
    print(f"  attn_scale_base: {output_data['best_result']['attn_scale_base']:.3f}")
    print(f"  attn_scale_coef: {output_data['best_result']['attn_scale_coef']:.3f}")
    print(f"  perplexity: {output_data['best_result']['perplexity']:.4f}")


def load_previous_results(filepath: str) -> List[Dict]:
    """
    Load results from previous search.

    Args:
        filepath: Path to previous results file

    Returns:
        List of previous results
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", [])


def run_grid_search(
    model,
    tokenizer,
    args,
    previous_results: List[Dict] = None,
) -> List[Dict]:
    """
    Run grid search over parameter space.

    Args:
        model: The language model
        tokenizer: The tokenizer
        args: Command line arguments
        previous_results: Results from previous run (for resuming)

    Returns:
        List of evaluation results
    """
    coef_values = generate_grid_search_space(
        args.attn_scale_coef_min,
        args.attn_scale_coef_max,
        args.attn_scale_coef_steps,
    )

    print(f"\n{'='*60}")
    print(f"Grid Search: {len(coef_values)} parameter combinations")
    print(f"{'='*60}")
    print(f"Base: 1.000 (fixed)")
    print(
        f"Coef range: [{args.attn_scale_coef_min:.3f}, {args.attn_scale_coef_max:.3f}]"
    )

    results = previous_results if previous_results else []
    evaluated_params = {r["attn_scale_coef"] for r in results}

    if previous_results:
        successful_prev = [r for r in previous_results if r.get("status") == "success"]
        if successful_prev:
            best_prev = min(successful_prev, key=lambda x: x["perplexity"])
            print(f"\n[Recovery] Resuming from previous results:")
            print(
                f"  Already evaluated: {len(evaluated_params)} parameter combinations"
            )
            print(
                f"  Current best: coef={best_prev['attn_scale_coef']:.3f}, ppl={best_prev['perplexity']:.4f}"
            )

    start_time = time.time()
    best_overall = None
    evaluations_count = 0

    for i, coef in enumerate(coef_values, 1):
        if coef in evaluated_params:
            print(
                f"[{i}/{len(coef_values)}] Skipping (already evaluated): coef={coef:.3f}"
            )
            continue

        print(f"\n[{i}/{len(coef_values)}] Evaluating: coef={coef:.3f}")

        try:
            eval_result = evaluate_params(
                model=model,
                tokenizer=tokenizer,
                attn_scale_coef=coef,
                eval_min_length=args.eval_min_length,
                eval_max_length=args.eval_max_length,
                dataset=args.eval_dataset,
                split=args.eval_split,
                limit=args.eval_limit,
                device=args.device,
            )

            result = {
                "attn_scale_base": 1.0,
                "attn_scale_coef": coef,
                "perplexity": eval_result["weighted_ppl"],
                "lengths": eval_result["lengths"],
                "perplexities": eval_result["perplexities"],
                "status": "success",
            }

            print(f"  ✓ Weighted Perplexity: {eval_result['weighted_ppl']:.4f}")
            evaluations_count += 1

            if (
                best_overall is None
                or eval_result["weighted_ppl"] < best_overall["perplexity"]
            ):
                best_overall = result.copy()
                print(f"  ★ New best found!")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            result = {
                "attn_scale_base": 1.0,
                "attn_scale_coef": coef,
                "perplexity": float("inf"),
                "status": "error",
                "error": str(e),
            }

        results.append(result)

        torch.cuda.empty_cache()
        gc.collect()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Grid Search Summary:")
    print(f"  Total evaluations: {evaluations_count}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    if best_overall:
        print(f"  Best attn_scale_base: {best_overall['attn_scale_base']:.3f}")
        print(f"  Best attn_scale_coef: {best_overall['attn_scale_coef']:.3f}")
        print(f"  Best perplexity: {best_overall['perplexity']:.4f}")
    print(f"{'='*60}")

    return results


def run_random_search(
    model,
    tokenizer,
    args,
    previous_results: List[Dict] = None,
) -> List[Dict]:
    """
    Run random search over parameter space.

    Args:
        model: The language model
        tokenizer: The tokenizer
        args: Command line arguments
        previous_results: Results from previous run (for resuming)

    Returns:
        List of evaluation results
    """
    coef_values = generate_random_search_space(
        args.attn_scale_coef_min,
        args.attn_scale_coef_max,
        args.random_samples,
    )

    print(f"\n{'='*60}")
    print(f"Random Search: {len(coef_values)} parameter combinations")
    print(f"{'='*60}")
    print(f"Base: 1.000 (fixed)")
    print(
        f"Coef range: [{args.attn_scale_coef_min:.3f}, {args.attn_scale_coef_max:.3f}]"
    )

    results = previous_results if previous_results else []
    evaluated_params = {r["attn_scale_coef"] for r in results}

    if previous_results:
        successful_prev = [r for r in previous_results if r.get("status") == "success"]
        if successful_prev:
            best_prev = min(successful_prev, key=lambda x: x["perplexity"])
            print(f"\n[Recovery] Resuming from previous results:")
            print(
                f"  Already evaluated: {len(evaluated_params)} parameter combinations"
            )
            print(
                f"  Current best: coef={best_prev['attn_scale_coef']:.3f}, ppl={best_prev['perplexity']:.4f}"
            )

    start_time = time.time()
    best_overall = None
    evaluations_count = 0

    for i, coef in enumerate(coef_values, 1):
        if coef in evaluated_params:
            print(
                f"[{i}/{len(coef_values)}] Skipping (already evaluated): coef={coef:.3f}"
            )
            continue

        print(f"\n[{i}/{len(coef_values)}] Evaluating: coef={coef:.3f}")

        try:
            eval_result = evaluate_params(
                model=model,
                tokenizer=tokenizer,
                attn_scale_coef=coef,
                eval_min_length=args.eval_min_length,
                eval_max_length=args.eval_max_length,
                dataset=args.eval_dataset,
                split=args.eval_split,
                limit=args.eval_limit,
                device=args.device,
            )

            result = {
                "attn_scale_base": 1.0,
                "attn_scale_coef": coef,
                "perplexity": eval_result["weighted_ppl"],
                "lengths": eval_result["lengths"],
                "perplexities": eval_result["perplexities"],
                "status": "success",
            }

            print(f"  ✓ Weighted Perplexity: {eval_result['weighted_ppl']:.4f}")
            evaluations_count += 1

            if (
                best_overall is None
                or eval_result["weighted_ppl"] < best_overall["perplexity"]
            ):
                best_overall = result.copy()
                print(f"  ★ New best found!")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            result = {
                "attn_scale_base": 1.0,
                "attn_scale_coef": coef,
                "perplexity": float("inf"),
                "status": "error",
                "error": str(e),
            }

        results.append(result)

        torch.cuda.empty_cache()
        gc.collect()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Random Search Summary:")
    print(f"  Total evaluations: {evaluations_count}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    if best_overall:
        print(f"  Best attn_scale_base: {best_overall['attn_scale_base']:.3f}")
        print(f"  Best attn_scale_coef: {best_overall['attn_scale_coef']:.3f}")
        print(f"  Best perplexity: {best_overall['perplexity']:.4f}")
    print(f"{'='*60}")

    return results


def run_bayesian_optimization(
    model,
    tokenizer,
    args,
    previous_results: List[Dict] = None,
) -> List[Dict]:
    """
    Run Bayesian optimization over parameter space.

    Args:
        model: The language model
        tokenizer: The tokenizer
        args: Command line arguments
        previous_results: Results from previous run (for resuming)

    Returns:
        List of evaluation results
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        print("Error: scipy is required for Bayesian optimization")
        print("Install with: pip install scipy")
        sys.exit(1)

    results = previous_results if previous_results else []

    def objective(params):
        coef = params[0]

        print(f"\nEvaluating: coef={coef:.3f}")

        try:
            eval_result = evaluate_params(
                model=model,
                tokenizer=tokenizer,
                attn_scale_coef=coef,
                eval_min_length=args.eval_min_length,
                eval_max_length=args.eval_max_length,
                dataset=args.eval_dataset,
                split=args.eval_split,
                limit=args.eval_limit,
                device=args.device,
            )

            result = {
                "attn_scale_base": 1.0,
                "attn_scale_coef": float(coef),
                "perplexity": eval_result["weighted_ppl"],
                "lengths": eval_result["lengths"],
                "perplexities": eval_result["perplexities"],
                "status": "success",
            }

            print(f"  ✓ Weighted Perplexity: {eval_result['weighted_ppl']:.4f}")
            perplexity = eval_result["weighted_ppl"]

        except Exception as e:
            print(f"  ✗ Error: {e}")
            perplexity = 1e10
            result = {
                "attn_scale_base": 1.0,
                "attn_scale_coef": float(coef),
                "perplexity": float("inf"),
                "status": "error",
                "error": str(e),
            }

        results.append(result)

        torch.cuda.empty_cache()
        gc.collect()

        return perplexity

    print(f"\n{'='*60}")
    print(f"Bayesian Optimization: {args.bayesian_iterations} iterations")
    print(f"{'='*60}")

    bounds = [(0.05, 0.3)]
    print(f"Base: 1.000 (fixed)")
    print(f"Coef bounds: [0.05, 0.3]")

    n_initial = min(10, args.bayesian_iterations // 3)
    print(f"Initial sampling: {n_initial} points")

    initial_points = []
    if previous_results:
        successful_prev = [r for r in previous_results if r.get("status") == "success"]
        if successful_prev:
            print(
                f"\n[Recovery] Found {len(successful_prev)} previous successful evaluations"
            )
            best_prev = min(successful_prev, key=lambda x: x["perplexity"])
            print(
                f"[Recovery] Best previous result: coef={best_prev['attn_scale_coef']:.3f}, ppl={best_prev['perplexity']:.4f}"
            )
    else:
        coef_points = torch.linspace(
            args.attn_scale_coef_min, args.attn_scale_coef_max, n_initial
        ).tolist()
        initial_points = [round(c, 3) for c in coef_points]

        print(f"\nPhase 1: Initial uniform sampling ({n_initial} points)")
        print(
            f"  Coef range: [{args.attn_scale_coef_min:.3f}, {args.attn_scale_coef_max:.3f}]"
        )
        print(f"  Points: {initial_points}")

        phase1_start = time.time()
        for i, coef in enumerate(initial_points, 1):
            print(f"\n[{i}/{n_initial}] Initial point: coef={coef:.3f}")
            objective([coef])

        phase1_elapsed = time.time() - phase1_start
        print(f"\nPhase 1 completed in {phase1_elapsed:.1f}s")

    if len(results) < n_initial and not previous_results:
        print("Warning: Not enough successful initial points for optimization")
        return results

    successful_results = [r for r in results if r["status"] == "success"]
    if not successful_results:
        print("Error: No successful evaluations")
        return results

    best_result = min(successful_results, key=lambda x: x["perplexity"])
    x0 = [best_result["attn_scale_coef"]]

    print(f"\n{'='*60}")
    print(f"Phase 2: Optimization from best initial point")
    print(f"  Starting point: coef={x0[0]:.3f}, ppl={best_result['perplexity']:.4f}")
    print(f"{'='*60}")

    remaining_iterations = args.bayesian_iterations - n_initial
    if previous_results:
        remaining_iterations = args.bayesian_iterations - len(previous_results)
        print(f"  Remaining iterations: {remaining_iterations}")

    best_overall = best_result.copy()
    no_improvement_count = 0
    max_no_improvement = 5

    for i in range(remaining_iterations):
        print(
            f"\n[{len(results) + 1}/{args.bayesian_iterations}] Optimization iteration {i + 1}"
        )

        try:
            result = minimize(
                objective,
                x0=x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 1, "disp": False},
            )

            if result.success:
                x0 = [round(result.x[0], 3)]
                successful_results = [r for r in results if r["status"] == "success"]
                if successful_results:
                    current_best = min(
                        successful_results, key=lambda x: x["perplexity"]
                    )

                    if current_best["perplexity"] < best_overall["perplexity"]:
                        best_overall = current_best.copy()
                        no_improvement_count = 0
                        print(
                            f"  ★ New best found: coef={best_overall['attn_scale_coef']:.3f}, ppl={best_overall['perplexity']:.4f}"
                        )
                    else:
                        no_improvement_count += 1
                        print(
                            f"  No improvement ({no_improvement_count}/{max_no_improvement})"
                        )

                    if no_improvement_count >= max_no_improvement:
                        print(
                            f"\n[Convergence] No improvement for {max_no_improvement} iterations"
                        )
                        print(f"[Convergence] Stopping early at iteration {i + 1}")
                        break

        except Exception as e:
            print(f"  Optimization error: {e}")
            successful_results = [r for r in results if r["status"] == "success"]
            if successful_results:
                best = min(successful_results, key=lambda x: x["perplexity"])
                x0 = [best["attn_scale_coef"]]

    print(f"\n{'='*60}")
    print(f"Bayesian Optimization Summary:")
    print(f"  Best attn_scale_base: 1.000 (fixed)")
    print(f"  Best attn_scale_coef: {best_overall['attn_scale_coef']:.3f}")
    print(f"  Best perplexity: {best_overall['perplexity']:.4f}")
    print(f"  Total evaluations: {len(results)}")
    print(f"{'='*60}")

    return results


def run_adaptive_search(
    model,
    tokenizer,
    args,
    previous_results: List[Dict] = None,
) -> List[Dict]:
    """
    Run adaptive multi-stage search over parameter space.

    This strategy performs multiple stages of search, progressively
    narrowing the search space around the best parameters found.

    Args:
        model: The language model
        tokenizer: The tokenizer
        args: Command line arguments
        previous_results: Results from previous run (for resuming)

    Returns:
        List of evaluation results
    """
    results = previous_results if previous_results else []

    center_coef = 0.15

    if previous_results:
        successful_prev = [r for r in previous_results if r.get("status") == "success"]
        if successful_prev:
            best_prev = min(successful_prev, key=lambda x: x["perplexity"])
            center_coef = best_prev["attn_scale_coef"]
            print(f"\n[Recovery] Resuming from best previous result:")
            print(
                f"  center_coef={center_coef:.3f}, perplexity={best_prev['perplexity']:.4f}"
            )

    center_coef = max(0.05, min(0.3, center_coef))

    radius_coef = (args.attn_scale_coef_max - args.attn_scale_coef_min) / 2

    best_overall = None
    convergence_threshold = 0.001
    prev_best_ppl = float("inf")

    for stage in range(args.adaptive_stages):
        print(f"\n{'='*60}")
        print(f"Adaptive Search - Stage {stage + 1}/{args.adaptive_stages}")
        print(f"{'='*60}")
        print(f"Base: 1.000 (fixed)")
        print(f"Search center: coef={center_coef:.3f}")
        print(f"Search radius: coef=±{radius_coef:.3f}")
        print(
            f"Coef range: [{max(0.05, center_coef - radius_coef):.3f}, {min(0.3, center_coef + radius_coef):.3f}]"
        )

        coef_min = max(0.05, center_coef - radius_coef)
        coef_max = min(0.3, center_coef + radius_coef)

        initial_samples = 30
        n_samples_coef = max(5, initial_samples - stage * 4)

        coef_values = torch.linspace(coef_min, coef_max, n_samples_coef).tolist()
        coef_values = [round(v, 3) for v in coef_values]

        print(
            f"Stage {stage + 1}: {len(coef_values)} parameter combinations to evaluate"
        )

        evaluated_params = {r["attn_scale_coef"] for r in results}

        stage_start_time = time.time()
        stage_results_count = 0

        for i, coef in enumerate(coef_values, 1):
            if coef in evaluated_params:
                print(
                    f"[{i}/{len(coef_values)}] Skipping (already evaluated): coef={coef:.3f}"
                )
                continue

            print(f"\n[Stage {stage+1}, {i}/{len(coef_values)}] coef={coef:.3f}")

            try:
                eval_result = evaluate_params(
                    model=model,
                    tokenizer=tokenizer,
                    attn_scale_coef=coef,
                    eval_min_length=args.eval_min_length,
                    eval_max_length=args.eval_max_length,
                    dataset=args.eval_dataset,
                    split=args.eval_split,
                    limit=args.eval_limit,
                    device=args.device,
                )

                result = {
                    "attn_scale_base": 1.0,
                    "attn_scale_coef": coef,
                    "perplexity": eval_result["weighted_ppl"],
                    "lengths": eval_result["lengths"],
                    "perplexities": eval_result["perplexities"],
                    "status": "success",
                    "stage": stage + 1,
                }

                print(f"  ✓ Weighted Perplexity: {eval_result['weighted_ppl']:.4f}")
                stage_results_count += 1

            except Exception as e:
                print(f"  ✗ Error: {e}")
                result = {
                    "attn_scale_base": 1.0,
                    "attn_scale_coef": coef,
                    "perplexity": float("inf"),
                    "status": "error",
                    "error": str(e),
                    "stage": stage + 1,
                }

            results.append(result)

            torch.cuda.empty_cache()
            gc.collect()

        successful_results = [r for r in results if r["status"] == "success"]
        if successful_results:
            best = min(successful_results, key=lambda x: x["perplexity"])
            center_coef = best["attn_scale_coef"]

            center_coef = max(0.05, min(0.3, center_coef))

            stage_elapsed = time.time() - stage_start_time
            print(f"\n{'='*60}")
            print(f"Stage {stage + 1} Summary:")
            print(f"  Best base: 1.000 (fixed)")
            print(f"  Best coef: {center_coef:.3f}")
            print(f"  Best perplexity: {best['perplexity']:.4f}")
            print(f"  Evaluations completed: {stage_results_count}")
            print(f"  Time elapsed: {stage_elapsed:.1f}s")
            print(f"{'='*60}")

            if best_overall is None or best["perplexity"] < best_overall["perplexity"]:
                best_overall = best.copy()

            improvement = prev_best_ppl - best["perplexity"]
            if stage > 0 and improvement < convergence_threshold:
                print(
                    f"\n[Convergence] Improvement ({improvement:.6f}) < threshold ({convergence_threshold})"
                )
                print(f"[Convergence] Search converged at stage {stage + 1}")
                break

            prev_best_ppl = best["perplexity"]

        radius_coef *= args.adaptive_refinement_factor

        if stage < args.adaptive_stages - 1 and successful_results:
            next_coef_min = max(0.05, center_coef - radius_coef)
            next_coef_max = min(0.3, center_coef + radius_coef)

            print(f"\n{'='*60}")
            print(f"Stage Transition: Stage {stage + 1} → Stage {stage + 2}")
            print(f"{'='*60}")
            print(f"  Current stage best parameters:")
            print(f"    attn_scale_base: 1.000 (fixed)")
            print(f"    attn_scale_coef: {center_coef:.3f}")
            print(f"  Current stage best perplexity: {best['perplexity']:.4f}")
            print(f"\n  Next stage search configuration:")
            print(f"    Search center: {center_coef:.3f}")
            print(f"    Search radius: ±{radius_coef:.3f}")
            print(f"    Search range: [{next_coef_min:.3f}, {next_coef_max:.3f}]")
            print(f"{'='*60}")

    if best_overall:
        print(f"\n{'='*60}")
        print(f"Overall Best Result:")
        print(f"  attn_scale_base: 1.000 (fixed)")
        print(f"  attn_scale_coef: {best_overall['attn_scale_coef']:.3f}")
        print(f"  perplexity: {best_overall['perplexity']:.4f}")
        print(f"  found at stage: {best_overall.get('stage', 'N/A')}")
        print(f"{'='*60}")

    return results


def run_log_scale_search(
    model,
    tokenizer,
    args,
    previous_results: List[Dict] = None,
) -> List[Dict]:
    """
    Run log-scale search over parameter space.

    This is useful when parameters span multiple orders of magnitude.

    Args:
        model: The language model
        tokenizer: The tokenizer
        args: Command line arguments
        previous_results: Results from previous run (for resuming)

    Returns:
        List of evaluation results
    """
    n_samples_coef = args.attn_scale_coef_steps

    coef_values = torch.logspace(
        math.log10(args.attn_scale_coef_min),
        math.log10(args.attn_scale_coef_max),
        n_samples_coef,
    ).tolist()
    coef_values = [round(v, 3) for v in coef_values]

    print(f"\n{'='*60}")
    print(f"Log-Scale Search: {len(coef_values)} parameter combinations")
    print(f"{'='*60}")
    print(f"Base: 1.000 (fixed)")
    print(f"Coef range: [{coef_values[0]:.3f}, {coef_values[-1]:.3f}]")
    print(
        f"  Log scale: log10({args.attn_scale_coef_min:.3f}) to log10({args.attn_scale_coef_max:.3f})"
    )

    results = previous_results if previous_results else []
    evaluated_params = {r["attn_scale_coef"] for r in results}

    if previous_results:
        successful_prev = [r for r in previous_results if r.get("status") == "success"]
        if successful_prev:
            best_prev = min(successful_prev, key=lambda x: x["perplexity"])
            print(f"\n[Recovery] Resuming from previous results:")
            print(
                f"  Already evaluated: {len(evaluated_params)} parameter combinations"
            )
            print(
                f"  Current best: coef={best_prev['attn_scale_coef']:.3f}, ppl={best_prev['perplexity']:.4f}"
            )

    start_time = time.time()
    best_overall = None
    evaluations_count = 0

    for i, coef in enumerate(coef_values, 1):
        if coef in evaluated_params:
            print(
                f"[{i}/{len(coef_values)}] Skipping (already evaluated): coef={coef:.3f}"
            )
            continue

        print(f"\n[{i}/{len(coef_values)}] Evaluating: coef={coef:.3f}")

        try:
            eval_result = evaluate_params(
                model=model,
                tokenizer=tokenizer,
                attn_scale_coef=coef,
                eval_min_length=args.eval_min_length,
                eval_max_length=args.eval_max_length,
                dataset=args.eval_dataset,
                split=args.eval_split,
                limit=args.eval_limit,
                device=args.device,
            )

            result = {
                "attn_scale_base": 1.0,
                "attn_scale_coef": coef,
                "perplexity": eval_result["weighted_ppl"],
                "lengths": eval_result["lengths"],
                "perplexities": eval_result["perplexities"],
                "status": "success",
            }

            print(f"  ✓ Weighted Perplexity: {eval_result['weighted_ppl']:.4f}")
            evaluations_count += 1

            if (
                best_overall is None
                or eval_result["weighted_ppl"] < best_overall["perplexity"]
            ):
                best_overall = result.copy()
                print(f"  ★ New best found!")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            result = {
                "attn_scale_base": 1.0,
                "attn_scale_coef": coef,
                "perplexity": float("inf"),
                "status": "error",
                "error": str(e),
            }

        results.append(result)

        torch.cuda.empty_cache()
        gc.collect()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Log-Scale Search Summary:")
    print(f"  Total evaluations: {evaluations_count}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    if best_overall:
        print(f"  Best attn_scale_base: 1.000 (fixed)")
        print(f"  Best attn_scale_coef: {best_overall['attn_scale_coef']:.3f}")
        print(f"  Best perplexity: {best_overall['perplexity']:.4f}")
    print(f"{'='*60}")

    return results


def run_bohb_search(
    model,
    tokenizer,
    args,
    previous_results: List[Dict] = None,
) -> List[Dict]:
    """
    Run BOHB (Bayesian Optimization + HyperBand) search.

    BOHB combines Bayesian optimization with HyperBand for efficient
    hyperparameter search with early stopping.

    Args:
        model: The language model
        tokenizer: The tokenizer
        args: Command line arguments
        previous_results: Results from previous run (for resuming)

    Returns:
        List of evaluation results
    """
    try:
        from scipy.optimize import minimize
        from scipy.stats import norm
    except ImportError:
        print("Error: scipy is required for BOHB optimization")
        print("Install with: pip install scipy")
        sys.exit(1)

    results = previous_results if previous_results else []

    low_fidelity_limit = max(1, args.eval_limit // args.bohb_early_stop_factor)
    high_fidelity_limit = args.eval_limit

    print(f"\n{'='*60}")
    print(f"BOHB Search: {args.bohb_iterations} iterations")
    print(f"{'='*60}")
    print(f"Base: 1.000 (fixed)")
    print(
        f"Coef range: [{args.attn_scale_coef_min:.3f}, {args.attn_scale_coef_max:.3f}]"
    )
    print(f"Low-fidelity samples: {low_fidelity_limit}")
    print(f"High-fidelity samples: {high_fidelity_limit}")
    print(f"Initial samples: {args.bohb_initial_samples}")

    def evaluate_with_fidelity(coef: float, fidelity: str = "low") -> Dict:
        """
        Evaluate parameter with specified fidelity level.

        Args:
            coef: Attention scale coefficient
            fidelity: "low" or "high" fidelity evaluation

        Returns:
            Evaluation result dictionary
        """
        limit = low_fidelity_limit if fidelity == "low" else high_fidelity_limit

        try:
            eval_result = evaluate_params(
                model=model,
                tokenizer=tokenizer,
                attn_scale_coef=coef,
                eval_min_length=args.eval_min_length,
                eval_max_length=args.eval_max_length,
                dataset=args.eval_dataset,
                split=args.eval_split,
                limit=limit,
                device=args.device,
            )

            result = {
                "attn_scale_base": 1.0,
                "attn_scale_coef": coef,
                "perplexity": eval_result["weighted_ppl"],
                "lengths": eval_result["lengths"],
                "perplexities": eval_result["perplexities"],
                "status": "success",
                "fidelity": fidelity,
            }

            return result

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return {
                "attn_scale_base": 1.0,
                "attn_scale_coef": coef,
                "perplexity": float("inf"),
                "status": "error",
                "error": str(e),
                "fidelity": fidelity,
            }

    start_time = time.time()

    if previous_results:
        successful_prev = [r for r in previous_results if r.get("status") == "success"]
        if successful_prev:
            best_prev = min(successful_prev, key=lambda x: x["perplexity"])
            print(f"\n[Recovery] Resuming from previous results:")
            print(f"  Already evaluated: {len(previous_results)} configurations")
            print(
                f"  Current best: coef={best_prev['attn_scale_coef']:.3f}, ppl={best_prev['perplexity']:.4f}"
            )
    else:
        print(f"\n{'='*60}")
        print(f"Phase 1: Initial Random Sampling ({args.bohb_initial_samples} samples)")
        print(f"  Using low-fidelity evaluation ({low_fidelity_limit} samples)")
        print(f"{'='*60}")

        phase1_start = time.time()
        initial_samples = generate_random_search_space(
            args.attn_scale_coef_min,
            args.attn_scale_coef_max,
            args.bohb_initial_samples,
        )

        for i, coef in enumerate(initial_samples, 1):
            print(
                f"\n[{i}/{args.bohb_initial_samples}] Initial sample: coef={coef:.3f}"
            )
            result = evaluate_with_fidelity(coef, fidelity="low")
            results.append(result)

            if result["status"] == "success":
                print(f"  ✓ Low-fidelity Perplexity: {result['perplexity']:.4f}")

            torch.cuda.empty_cache()
            gc.collect()

        phase1_elapsed = time.time() - phase1_start
        print(f"\nPhase 1 completed in {phase1_elapsed:.1f}s")

    successful_results = [r for r in results if r["status"] == "success"]
    if not successful_results:
        print("Error: No successful evaluations in initial sampling")
        return results

    print(f"\n{'='*60}")
    print(f"Phase 2: HyperBand Early Stopping")
    print(f"{'='*60}")

    phase2_start = time.time()
    successful_results_sorted = sorted(
        successful_results, key=lambda x: x["perplexity"]
    )
    top_k = max(1, len(successful_results_sorted) // 3)
    top_configs = successful_results_sorted[:top_k]

    print(f"  Total initial configs: {len(successful_results)}")
    print(f"  Top configs to promote: {len(top_configs)} (top 1/3)")

    high_fidelity_results = []
    for i, config in enumerate(top_configs, 1):
        coef = config["attn_scale_coef"]
        print(f"\n[{i}/{len(top_configs)}] High-fidelity evaluation: coef={coef:.3f}")

        result = evaluate_with_fidelity(coef, fidelity="high")
        high_fidelity_results.append(result)
        results.append(result)

        if result["status"] == "success":
            print(f"  ✓ High-fidelity Perplexity: {result['perplexity']:.4f}")

        torch.cuda.empty_cache()
        gc.collect()

    phase2_elapsed = time.time() - phase2_start
    print(f"\nPhase 2 completed in {phase2_elapsed:.1f}s")

    if not high_fidelity_results or all(
        r["status"] != "success" for r in high_fidelity_results
    ):
        print("Warning: No successful high-fidelity evaluations")
        best_overall = min(successful_results, key=lambda x: x["perplexity"])
    else:
        successful_high_fidelity = [
            r for r in high_fidelity_results if r["status"] == "success"
        ]
        best_overall = min(successful_high_fidelity, key=lambda x: x["perplexity"])

    print(f"\n{'='*60}")
    print(
        f"Phase 3: Bayesian Optimization Iterations ({args.bohb_iterations} iterations)"
    )
    print(f"{'='*60}")

    phase3_start = time.time()

    def acquisition_function(
        coef: float, explored_params: List[float], explored_values: List[float]
    ) -> float:
        """
        Expected Improvement acquisition function.

        Args:
            coef: Parameter to evaluate
            explored_params: Previously explored parameters
            explored_values: Corresponding objective values

        Returns:
            Expected improvement value
        """
        if len(explored_params) < 2:
            return 0.0

        best_value = min(explored_values)

        mean = np.mean(explored_values)
        std = np.std(explored_values) + 1e-9

        distances = [abs(coef - p) for p in explored_params]
        min_dist = min(distances)

        exploration_bonus = -min_dist * 0.1

        z = (best_value - mean) / std
        ei = (best_value - mean) * norm.cdf(z) + std * norm.pdf(z)

        return ei + exploration_bonus

    explored_coefs = [r["attn_scale_coef"] for r in successful_results]
    explored_ppls = [r["perplexity"] for r in successful_results]

    no_improvement_count = 0
    max_no_improvement = 10

    for iteration in range(args.bohb_iterations):
        print(f"\n[Iteration {iteration + 1}/{args.bohb_iterations}]")

        best_current = min(
            [r for r in results if r["status"] == "success"],
            key=lambda x: x["perplexity"],
        )
        x0 = [best_current["attn_scale_coef"]]

        def objective(params):
            coef = params[0]
            coef = max(args.attn_scale_coef_min, min(args.attn_scale_coef_max, coef))

            for i, existing_coef in enumerate(explored_coefs):
                if abs(coef - existing_coef) < 0.001:
                    return explored_ppls[i] + 100

            result = evaluate_with_fidelity(round(coef, 3), fidelity="low")
            results.append(result)

            if result["status"] == "success":
                explored_coefs.append(result["attn_scale_coef"])
                explored_ppls.append(result["perplexity"])

            torch.cuda.empty_cache()
            gc.collect()

            return result["perplexity"] if result["status"] == "success" else 1e10

        try:
            opt_result = minimize(
                objective,
                x0=x0,
                method="L-BFGS-B",
                bounds=[(args.attn_scale_coef_min, args.attn_scale_coef_max)],
                options={"maxiter": 1, "disp": False},
            )

            if opt_result.success:
                proposed_coef = round(opt_result.x[0], 3)
                print(f"  Proposed coef: {proposed_coef:.3f}")

                successful_current = [r for r in results if r["status"] == "success"]
                if successful_current:
                    current_best = min(
                        successful_current, key=lambda x: x["perplexity"]
                    )

                    if current_best["perplexity"] < best_overall["perplexity"]:
                        best_overall = current_best.copy()
                        no_improvement_count = 0
                        print(
                            f"  ★ New best found: coef={best_overall['attn_scale_coef']:.3f}, ppl={best_overall['perplexity']:.4f}"
                        )
                    else:
                        no_improvement_count += 1

                    if no_improvement_count >= max_no_improvement:
                        print(
                            f"\n[Convergence] No improvement for {max_no_improvement} iterations"
                        )
                        print(
                            f"[Convergence] Stopping early at iteration {iteration + 1}"
                        )
                        break

        except Exception as e:
            print(f"  Optimization error: {e}")

        if (iteration + 1) % 10 == 0:
            print(
                f"\n[Checkpoint] High-fidelity validation at iteration {iteration + 1}"
            )
            successful_current = [r for r in results if r["status"] == "success"]
            if successful_current:
                current_best = min(successful_current, key=lambda x: x["perplexity"])
                coef = current_best["attn_scale_coef"]

                already_high_fidelity = any(
                    r["attn_scale_coef"] == coef and r.get("fidelity") == "high"
                    for r in results
                )

                if not already_high_fidelity:
                    print(
                        f"  Validating best config with high-fidelity: coef={coef:.3f}"
                    )
                    result = evaluate_with_fidelity(coef, fidelity="high")
                    results.append(result)

                    if result["status"] == "success":
                        print(
                            f"  ✓ High-fidelity Perplexity: {result['perplexity']:.4f}"
                        )
                        if result["perplexity"] < best_overall["perplexity"]:
                            best_overall = result.copy()
                            print(f"  ★ New best with high-fidelity!")

                    torch.cuda.empty_cache()
                    gc.collect()

    phase3_elapsed = time.time() - phase3_start
    print(f"\nPhase 3 completed in {phase3_elapsed:.1f}s")

    print(f"\n{'='*60}")
    print(f"Final High-Fidelity Validation")
    print(f"{'='*60}")

    successful_current = [r for r in results if r["status"] == "success"]
    if successful_current:
        final_best = min(successful_current, key=lambda x: x["perplexity"])
        coef = final_best["attn_scale_coef"]

        already_high_fidelity = any(
            r["attn_scale_coef"] == coef and r.get("fidelity") == "high"
            for r in results
        )

        if not already_high_fidelity:
            print(f"  Final validation: coef={coef:.3f}")
            result = evaluate_with_fidelity(coef, fidelity="high")
            results.append(result)

            if result["status"] == "success":
                print(f"  ✓ Final High-fidelity Perplexity: {result['perplexity']:.4f}")
                if result["perplexity"] < best_overall["perplexity"]:
                    best_overall = result.copy()

            torch.cuda.empty_cache()
            gc.collect()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"BOHB Search Summary:")
    print(f"  Total evaluations: {len(results)}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    print(f"  Best attn_scale_base: 1.000 (fixed)")
    print(f"  Best attn_scale_coef: {best_overall['attn_scale_coef']:.3f}")
    print(f"  Best perplexity: {best_overall['perplexity']:.4f}")
    print(f"{'='*60}")

    return results


def main():
    args = parse_args()

    print("=" * 60)
    print("Attention Scale Parameter Search")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"RoPE type: {args.rope_type}")
    print(f"Search method: {args.search_method}")
    print(f"Evaluation length range: {args.eval_min_length} - {args.eval_max_length}")
    print(f"Dataset: {args.eval_dataset}")
    print(f"Output directory: {args.output_dir}")

    if args.search_method == "adaptive":
        print(f"Adaptive stages: {args.adaptive_stages}")
        print(f"Refinement factor: {args.adaptive_refinement_factor}")

    if args.search_method == "bohb":
        print(f"BOHB initial samples: {args.bohb_initial_samples}")
        print(f"BOHB iterations: {args.bohb_iterations}")
        print(f"BOHB early stop factor: {args.bohb_early_stop_factor}")

    print("=" * 60)
    print(f"\nParameter search space:")
    print(f"  attn_scale_base: 1.000 (fixed)")
    print(
        f"  attn_scale_coef: [{args.attn_scale_coef_min:.3f}, {args.attn_scale_coef_max:.3f}]"
    )
    print("=" * 60)

    previous_results = None
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        previous_results = load_previous_results(args.resume)
        print(f"Loaded {len(previous_results)} previous results")

    print("\nLoading model...")
    model, config = load_model(args)
    tokenizer = load_tokenizer(args)

    print(f"Model loaded successfully")
    print(f"Model device: {next(model.parameters()).device}")

    if args.search_method == "grid":
        results = run_grid_search(model, tokenizer, args, previous_results)
    elif args.search_method == "random":
        results = run_random_search(model, tokenizer, args, previous_results)
    elif args.search_method == "bayesian":
        results = run_bayesian_optimization(model, tokenizer, args, previous_results)
    elif args.search_method == "adaptive":
        results = run_adaptive_search(model, tokenizer, args, previous_results)
    elif args.search_method == "log-scale":
        results = run_log_scale_search(model, tokenizer, args, previous_results)
    elif args.search_method == "bohb":
        results = run_bohb_search(model, tokenizer, args, previous_results)
    else:
        raise ValueError(f"Unknown search method: {args.search_method}")

    save_results(results, args.output_dir, args)

    print("\n" + "=" * 60)
    print("Search completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
