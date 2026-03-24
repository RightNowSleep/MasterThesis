import argparse
import json
import os
import lm_eval
from lm_eval.tasks import TaskManager
from lm_eval.models.huggingface import HFLM
import warnings
import transformers

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

from models.model_loader import load_model, load_tokenizer, add_args_model


# ---------------------------------------------------------------------------
# Default few-shot counts that mirror the yarn eval_harness.sh conventions
# ---------------------------------------------------------------------------
_DEFAULT_FEWSHOT = {
    "arc_challenge": 25,
    "hellaswag": 10,
    "truthfulqa_mc1": 0,
    "truthfulqa_mc2": 0,
    "mmlu": 5,
    "longbench2": 0,
}

# ---------------------------------------------------------------------------
# Task name validation helper (optional; avoids cryptic lm-eval errors)
# ---------------------------------------------------------------------------


def _warn_unknown_tasks(task_list: list[str]) -> None:
    """Print a warning for any task that isn't in lm-eval's registry."""
    try:
        tm = TaskManager()
        all_tasks = set(tm.all_tasks)
        for t in task_list:
            if t not in all_tasks:
                print(
                    f"[WARNING] Task '{t}' is not in lm-eval's registry.  "
                    "Check `lm-eval --tasks list` for valid names."
                )
    except Exception:
        pass  # Non-fatal; lm-eval will catch unknown tasks itself


# ---------------------------------------------------------------------------
# Output filename
# ---------------------------------------------------------------------------


def _build_output_path(output_dir: str, model_name: str, config) -> str:
    """Build a descriptive output JSON filename from model + RoPE config."""
    model_label = model_name.rstrip("/").split("/")[-1]
    rope_scaling = config.rope_scaling
    rope_label = rope_scaling["type"] if rope_scaling else "none"
    if rope_label != "none":
        factor = rope_scaling.get("factor", None)
        dynamic = rope_scaling.get("dynamic", False)
        if factor is not None:
            rope_label += f"_factor{str(factor).replace('.', '_')}"
        elif dynamic:
            rope_label += "_dynamic"
    filename = f"{model_label}_{rope_label}.json"
    return os.path.join(output_dir, filename)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    # ── 0. Validate tasks ─────────────────────────────────────────────── #
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    _warn_unknown_tasks(task_list)

    # ── 1. Load model with our loader (correct RoPE) ──────────────────── #
    # Training scripts set use_cache=False; here we want it True for
    # lm-eval's generation tasks (generate_until).  add_args_model already
    # defaults use_cache=True.
    model, config = load_model(args)
    tokenizer = load_tokenizer(args)
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling["type"] if rope_scaling else "none"
    rope_factor = rope_scaling.get("factor", None) if rope_scaling else "None"
    rope_dynamic = rope_scaling.get("dynamic", False) if rope_scaling else "False"
    print(f"\n{'='*60}")
    print(f"Loading model: {args.model_name}")
    print(f"RoPE type    : {rope_type}")
    print(f"RoPE factor  : {rope_factor}")
    print(f"RoPE dynamic : {rope_dynamic}")
    print(f"Max length   : {args.max_length}")
    print(f"Tasks        : {', '.join(task_list)}")
    print(f"{'='*60}\n")

    # ── 2. Wrap in lm-eval's HFLM ─────────────────────────────────────── #
    # HFLM accepts a pre-loaded model when pretrained is a PreTrainedModel
    # instance.  This bypasses AutoModelForCausalLM entirely, so our custom
    # RoPE classes (registered in models/pe_llama.py) are used as-is.
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    # ── 3. Build per-task few-shot counts ─────────────────────────────── #
    if args.num_fewshot is not None:
        # Single value overrides everything
        num_fewshot = args.num_fewshot
    else:
        # Use task-specific defaults where defined, else 0
        num_fewshot = {t: _DEFAULT_FEWSHOT.get(t, 0) for t in task_list}

    # ── 4. Run evaluation ─────────────────────────────────────────────── #
    results = {}
    out_path = _build_output_path(args.output_dir, args.model_name, config)

    for task in task_list:
        num_fewshot_task = num_fewshot.get(task, 0)
        print(f"\n{'='*60}")
        print(f"Running task: {task} (few-shot: {num_fewshot_task})")
        print(f"{'='*60}")

        result = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task],
            num_fewshot=num_fewshot_task,
            batch_size=args.batch_size,
            log_samples=args.log_samples,
        )
        results[task] = result

        # ── 5. Save results after each task ─────────────────────────────── #
        results_to_save = dict(results)
        results_to_save["metadata"] = {
            "model_name": args.model_name,
            "adapter_path": args.adapter_path,
            "rope_type": rope_type,
            "rope_factor": rope_factor,
            "rope_dynamic": rope_dynamic,
            "max_length": args.max_length,
            "tasks": task_list,
            "completed_tasks": list(results.keys()),
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[Checkpoint] Results saved after task '{task}' → {out_path}")

        # Print task result immediately
        if result and "results" in result:
            print(f"\n  Task '{task}' results:")
            for metric, value in result["results"].get(task, {}).items():
                if isinstance(value, float):
                    print(f"    {metric}: {value:.4f}  ({value*100:.2f}%)")
                elif not metric.startswith("_"):
                    print(f"    {metric}: {value}")

    # ── 6. Print final summary ────────────────────────────────────────── #
    print(f"\n{'='*60}")
    print(f"ALL TASKS [{', '.join(task_list)}] COMPLETED - FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total tasks completed: {len(results)}")
    print(f"Results saved → {out_path}")
    print(f"{'='*60}\n")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_args_harness(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add lm-eval-harness-specific arguments."""
    parser.add_argument(
        "--tasks",
        type=str,
        default="longbench2,arc_challenge,hellaswag,truthfulqa_mc1,mmlu",
        help=(
            "Comma-separated list of lm-eval task names.  "
            "Examples: longbench2, arc_challenge, hellaswag, "
            "truthfulqa_mc1, mmlu.  "
            "Run `python -m lm_eval --tasks list` for all available tasks."
        ),
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help=(
            "Number of few-shot examples for all tasks.  "
            "If omitted, uses per-task defaults: "
            "arc_challenge=25, hellaswag=10, truthfulqa=0, mmlu=5, "
            "longbench2=0."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for lm-eval inference.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/harness",
        help="Directory for saving evaluation results.",
    )
    parser.add_argument(
        "--log-samples",
        action="store_true",
        help="Save per-sample predictions alongside aggregate results.",
    )
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run lm-evaluation-harness with MasterThesis custom RoPE models.  "
            "Loads the model via models/model_loader.py (correct RoPE config), "
            "then passes the live model instance to lm_eval.simple_evaluate()."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser = add_args_model(parser)
    parser = add_args_harness(parser)
    args = parser.parse_args()
    main(args)
