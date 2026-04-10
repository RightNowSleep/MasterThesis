"""
eval/eval_harness.py
---------------------
lm-evaluation-harness runner for MasterThesis custom RoPE models.

This module wraps the lm_eval library to run standard NLP benchmarks (long-context,
reasoning, math, code) on LLaMA-style models with custom Rotary Position Embedding
(RoPE) configurations. It loads models via the project's model_loader (which applies
the correct RoPE scaling), wraps them in lm-eval's HFLM, and runs evaluation tasks
with checkpoint resume support.

Key features:
    - Task categorisation: long_context, reasoning, math, code.
    - Per-task default few-shot counts mirroring yarn eval_harness.sh conventions.
    - Error logging with automatic resume: failed tasks are skipped on re-run.
    - Results saved per-task as JSON with full metadata.

Usage:
    python eval/eval_harness.py --model-name huggyllama/llama-7b --tasks arc_challenge,mmlu
"""

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
    "niah": [  # negative
        "niah_single_1",
        "niah_single_2",
        "niah_single_3",
        "niah_multikey_1",
        "niah_multikey_2",
        "niah_multikey_3",
        "niah_multiquery",
        "niah_multivalue",
    ],
    "long_context": [
        "longbench",  # negative
        "longbench2",
        "longcxt",  # too long
        "passkey",  # negative
        "ruler",  # negative
        "babilong",
    ],
    "reasoning": [
        "arc_challenge",
        "truthfulqa",
        "hellaswag",
        "triviaqa",
        "bbh",  # too long
        "mmlu",
        "babi",
        "bbq",
    ],
    "math": [
        "gsm8k",
        "aime",  # 32K
        "hendrycks_math",  # negative
        "hendrycks_math500",  # negative
        "aime24",  # 32K
        "aime25",  # 32K
        "asdiv",  # negative
        "arithmetic",  # negative
        "math_word_problems",  # negative
        "hrm8k",  # negative
        "agieval_math",
        "minerva_math",
    ],
    "code": [
        "humaneval",  # negative
        "mbpp",
        "code2text",  # too long
    ],
}

_METADATA = {"max_seq_lengths": [2048, 4096, 8192, 16384]}
_ERROR_LOG_FILE = "error_log.json"

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
# Error log persistence (checkpoint resume support)
# ---------------------------------------------------------------------------


def _load_error_log(output_dir: str) -> dict:
    """
    Load existing error log from disk.

    Args:
        output_dir: Base output directory.

    Returns:
        Dict mapping task name → error info dict.
    """
    path = os.path.join(output_dir, _ERROR_LOG_FILE)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_error_log(output_dir: str, error_log: dict) -> None:
    """
    Save error log to disk (atomic write).

    Args:
        output_dir: Base output directory.
        error_log: Dict mapping task name → error info dict.
    """
    path = os.path.join(output_dir, _ERROR_LOG_FILE)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(error_log, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_path, path)


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
    """
    Run the full lm-evaluation-harness pipeline.

    Loads the model with correct RoPE configuration, wraps it in HFLM, iterates
    over the requested task list, and saves per-task results to JSON. Failed
    tasks are logged to an error-log file so they can be skipped on subsequent
    runs (checkpoint resume).

    Args:
        args: Parsed argparse.Namespace containing all CLI arguments
            (model_name, tasks, output_dir, batch_size, limit, etc.).

    Returns:
        None. Results are written to disk as JSON files under args.output_dir.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    # ── 0. Load error log & filter already-failed tasks ──────────────── #
    error_log = _load_error_log(args.output_dir)
    if error_log:
        print(f"\n[Resume] Loaded error log with {len(error_log)} recorded failure(s):")
        for task, info in error_log.items():
            print(f"  - {task}: {info.get('error', 'unknown')[:100]}")

    raw_task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    _warn_unknown_tasks(raw_task_list)

    # Filter out tasks that already failed in previous runs
    skipped_tasks = [t for t in raw_task_list if t in error_log]
    task_list = [t for t in raw_task_list if t not in error_log]
    if skipped_tasks:
        print(
            f"\n[Skip] Skipping {len(skipped_tasks)} previously-failed task(s): {', '.join(skipped_tasks)}"
        )
        print(
            f"        Use --tasks to specify only the tasks you want to retry, or delete {os.path.join(args.output_dir, _ERROR_LOG_FILE)} to reset."
        )
    if not task_list:
        print(
            "\n[Info] All tasks have been previously recorded as failures. Nothing to run."
        )
        return

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
    new_errors = 0

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
            try:
                result = lm_eval.simple_evaluate(
                    model=lm,
                    tasks=[task],
                    num_fewshot=num_fewshot_task,
                    batch_size=args.batch_size,
                    log_samples=args.log_samples,
                    limit=args.limit,
                    confirm_run_unsafe_code=True,
                    metadata=_METADATA,
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
                    json.dump(
                        results_to_save,
                        f,
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )
                print(
                    f"\n[Checkpoint] Results saved after task '{task}' → {output_path}"
                )

                # Print task result immediately
                if result and "results" in result:
                    print(f"\n  Task '{task}' results:")
                    for metric, value in result["results"].get(task, {}).items():
                        if isinstance(value, float):
                            print(f"    {metric}: {value:.4f}  ({value*100:.2f}%)")
                        elif not metric.startswith("_"):
                            print(f"    {metric}: {value}")
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                ERRORS[task] = error_msg
                error_log[task] = {
                    "error": error_msg,
                    "type": error_type,
                    "model_name": args.model_name,
                    "rope_type": rope_type,
                    "timestamp": __import__("datetime").datetime.now().isoformat(),
                }
                _save_error_log(args.output_dir, error_log)
                new_errors += 1
                print(f"Error running task '{task}': [{error_type}] {error_msg}")
                print(
                    f"  → Error logged to {os.path.join(args.output_dir, _ERROR_LOG_FILE)}"
                )

    # ── 5. Print final summary ────────────────────────────────────────── #
    all_tasks_raw = raw_task_list
    print(f"\n{'='*60}")
    print(f"ALL TASKS [{', '.join(all_tasks_raw)}] EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Tasks attempted this run : {len(task_list)}")
    print(f"Tasks completed          : {len(task_list) - len(ERRORS)}")
    print(f"New errors this run      : {new_errors}")
    print(f"Previously skipped      : {len(skipped_tasks)}")
    print(f"Total errors on record   : {len(error_log)}")
    print(
        f"Error log file           : {os.path.join(args.output_dir, _ERROR_LOG_FILE)}"
    )
    print(f"{'='*60}\n")
    if ERRORS:
        print(f"New errors this run:")
        print(f"{'='*60}\n")
        for task, error in ERRORS.items():
            info = error_log.get(task, {})
            print(f"  [{info.get('type', 'Exception')}] {task}:")
            print(f"    {error[:200]}{'...' if len(error) > 200 else ''}")


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
