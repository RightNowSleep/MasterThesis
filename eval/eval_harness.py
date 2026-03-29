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
    "bbh": 3,
    "mmlu": 5,
    "gsm8k": 8,
    "hendrycks_math": 4,
}

_TASKS_MAP = {
    "long_context": [
        "longbench",
        "longbench2",
        "longcxt",
        "passkey",
        "ruler",
        "babilong",
    ],
    "reasoning": [
        "arc_challenge",
        "truthfulqa",
        "hellaswag",
        "bbh",
        "mmlu",
    ],
    "math": [
        "gsm8k",
        "aime",
        "hendrycks_math",
    ],
    "code": [
        "humaneval",
        "mbpp",
        "humaneval_infilling",
    ],
}

_METADATA = {"max_seq_lengths": [2048, 4096, 8192, 16384]}

# ---------------------------------------------------------------------------
# Task name validation helper (optional; avoids cryptic lm-eval errors)
# ---------------------------------------------------------------------------


def _warn_unknown_tasks(task_list: list[str]) -> None:
    """
    Print a warning for any task that isn't in lm-eval's registry.

    Args:
        task_list: List of task names to validate.
    """
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
# Output
# ---------------------------------------------------------------------------


def _build_output_dir(task: str, output_dir: str) -> str:
    """
    Build a directory for saving task results.

    Args:
        task: Task name.
        output_dir: Base output directory.

    Returns:
        Full path to the task-specific output directory.
    """
    dir = output_dir
    for category in _TASKS_MAP:
        if task in _TASKS_MAP[category]:
            dir = os.path.join(output_dir, category)
            break
    dir = os.path.join(dir, task)
    os.makedirs(dir, exist_ok=True)
    return dir


def _build_output_path(output_dir: str, model_name: str, config) -> str:
    """
    Build a descriptive output JSON filename from model + RoPE config.

    Args:
        output_dir: Output directory path.
        model_name: Name of the model.
        config: Model configuration object.

    Returns:
        Full path to the output JSON file.
    """
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
    _METADATA["pretrained"] = args.model_name
    ERRORS = dict()

    # ── 3. Run evaluation ─────────────────────────────────────────────── #
    for task in task_list:
        try:
            output_dir = _build_output_dir(task, args.output_dir)
            output_path = _build_output_path(output_dir, args.model_name, config)
            num_fewshot_task = _DEFAULT_FEWSHOT.get(task, None)
            print(f"\\n{'='*60}")
            print(f"Running task: {task} (few-shot: {num_fewshot_task})")
            print(f"{'='*60}")

            result = lm_eval.simple_evaluate(
                model=lm,
                tasks=[task],
                num_fewshot=num_fewshot_task,
                batch_size=args.batch_size,
                log_samples=args.log_samples,
                limit=args.limit,
                confirm_run_unsafe_code=True,
            )

            # ── 4. Save results after each task ─────────────────────────────── #
            results_to_save = dict(result)
            results_to_save["metadata"] = {
                "model_name": args.model_name,
                "adapter_path": args.adapter_path,
                "rope_type": rope_type,
                "rope_factor": rope_factor,
                "rope_dynamic": rope_dynamic,
                "max_length": args.max_length,
                "task": task,
                "completed_tasks": task,
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results_to_save, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n[Checkpoint] Results saved after task '{task}' → {output_path}")

            # Print task result immediately
            if result and "results" in result:
                print(f"\n  Task '{task}' results:")
                for metric, value in result["results"].get(task, {}).items():
                    if isinstance(value, float):
                        print(f"    {metric}: {value:.4f}  ({value*100:.2f}%)")
                    elif not metric.startswith("_"):
                        print(f"    {metric}: {value}")
        except Exception as e:
            ERRORS[task] = str(e)
            print(f"Error running task '{task}': {e}")

    # ── 5. Print final summary ────────────────────────────────────────── #
    print(f"\n{'='*60}")
    print(f"ALL TASKS [{', '.join(task_list)}] COMPLETED - FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total tasks completed: {len(task_list)}")
    print(f"Results saved → {output_path}")
    print(f"{'='*60}\n")
    if ERRORS:
        print(f"Errors encountered: {', '.join(ERRORS.keys())}")
        print(f"{'='*60}\n")
        for task, error in ERRORS.items():
            print(f"  {task}: {error}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_args_harness(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add lm-eval-harness-specific arguments.

    Args:
        parser: ArgumentParser to add arguments to.

    Returns:
        Parser with added arguments.
    """
    parser.add_argument(
        "--tasks",
        type=str,
        default=(
            "longbench,longbench2,longcxt,passkey,ruler,babilong,arc_challenge,truthfulqa,hellaswag,"
            "bbh,mmlu,gsm8k,aime,hendrycks_math,humaneval,mbpp,humaneval_infilling"
        ),
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
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples for each task to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for saving evaluation results.",
    )
    parser.add_argument(
        "--log-samples",
        action="store_true",
        default=False,
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
