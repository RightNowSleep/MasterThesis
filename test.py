"""Unified evaluation test runner for context-extended language models.

Provides a CLI entry point that dispatches to one of four evaluation backends:

    - **perplexity**: Length-stratified perplexity evaluation across multiple
      context lengths to measure language modeling quality after extension.
    - **passkey**: Passkey retrieval test measuring the model's ability to find
      a hidden key at various positions within a long context.
    - **quality**: Multi-task benchmark evaluation (e.g., MMLU, ARC, Hellaswag)
      assessing downstream task performance.
    - **performance**: Inference profiling measuring throughput (tokens/sec) and
      GPU memory consumption across increasing sequence lengths.

Each subcommand accepts its own set of hyperparameters and produces a JSON result
file named automatically based on the model name and RoPE configuration::

    llama-7b_linear_factor4_0.json
"""

import argparse
import os

os.environ["USE_FLASH_ATTN"] = "0"

from models.model_loader import add_args_model, load_model, load_tokenizer
from eval.perplexity import PerplexityEvaluator, add_args_perplexity
from eval.passkey import PasskeyEvaluator, add_args_passkey
from eval.quality import QualityEvaluator, add_args_quality
from eval.performance import PerformanceEvaluator, add_args_performance


def generate_save_filename(model_name, config):
    """Generate a deterministic result filename from model name and RoPE config.

    Encodes the RoPE scaling type and parameters into the filename so that results
    from different configurations never overwrite each other.

    Args:
        model_name: Full HuggingFace model identifier (e.g.,
            ``"meta-llama/Llama-2-7b-hf"``).
        config: Model configuration object with a ``rope_scaling`` attribute.

    Returns:
        str: Filename string without directory path, ending in ``.json``.

    Examples:
        >>> generate_save_filename("meta-llama/Llama-2-7b-hf", cfg_none)
        'llama-7b_none.json'
        >>> generate_save_filename("meta-llama/Llama-2-7b-hf", cfg_dyn)
        'llama-7b_linear_dynamic.json'
        >>> generate_save_filename("meta-llama/Llama-2-7b-hf", cfg_fac)
        'llama-7b_linear_factor4_0.json'
    """
    model_name = model_name.split("/")[-1]
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling["type"] if rope_scaling else "none"
    parts = [model_name, rope_type]
    if rope_type != "none":
        factor = rope_scaling.get("factor", None)
        dynamic = rope_scaling.get("dynamic", False)
        if factor is not None:
            parts.append(f"factor{str(factor).replace('.', '_')}")
        elif dynamic:
            parts.append("dynamic")
    return "_".join(parts) + ".json"


def test_perplexity(args):
    """Run perplexity evaluation and print results.

    Loads the model and tokenizer, instantiates a :class:`PerplexityEvaluator`
    with parameters drawn from *args*, runs the evaluation, and prints the
    resulting metrics dictionary.

    Args:
        args: Parsed CLI arguments containing model, dataset, and perplexity
            evaluation settings.

    Returns:
        None
    """
    model, config = load_model(args)
    tokenizer = load_tokenizer(args)
    args.save_file = args.save_file or generate_save_filename(args.model_name, config)
    evaluator = PerplexityEvaluator(
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
        sliding_window=args.sliding_window,
        truncate=args.truncate,
        aggressive_memory=args.aggressive_memory,
        save_dir=args.save_dir,
        save_file=args.save_file,
    )
    results = evaluator.evaluate()
    print(results)


def test_passkey(args):
    """Run passkey retrieval evaluation and print results.

    Loads the model and tokenizer, instantiates a :class:`PasskeyEvaluator`,
    runs the passkey test across multiple context lengths, and prints the results.

    Args:
        args: Parsed CLI arguments containing model, passkey-specific parameters
            (number of keys, iterations, data mode, etc.), and output settings.

    Returns:
        None
    """
    model, config = load_model(args)
    tokenizer = load_tokenizer(args)
    args.save_file = args.save_file or generate_save_filename(args.model_name, config)
    evaluator = PasskeyEvaluator(
        model=model,
        tokenizer=tokenizer,
        restrict_tokens=args.restrict_tokens,
        data_mode=args.data_mode,
        dataset_name=args.dataset_name,
        split=args.split,
        min_length=args.min_length,
        max_length=args.max_length,
        aggressive_memory=args.aggressive_memory,
        num_keys=args.num_keys,
        iterations=args.iterations,
        length_step=args.length_step,
        save_dir=args.save_dir,
        save_file=args.save_file,
    )
    results = evaluator.evaluate()
    print(results)


def test_quality(args):
    """Run quality benchmark evaluation and print results.

    Loads the model and tokenizer, instantiates a :class:`QualityEvaluator` for
    multi-task assessment, runs the evaluation, and prints the results.

    Args:
        args: Parsed CLI arguments containing model, benchmark subset selection,
            chain-of-thought options, and output settings.

    Returns:
        None
    """
    model, config = load_model(args)
    tokenizer = load_tokenizer(args)
    args.save_file = args.save_file or generate_save_filename(args.model_name, config)
    evaluator = QualityEvaluator(
        model=model,
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        subset=args.subset,
        split=args.split,
        limit=args.limit,
        max_length=args.max_length,
        cot=args.cot,
        no_context=args.no_context,
        rag=args.rag,
        aggressive_memory=args.aggressive_memory,
        scoring_mode=args.scoring_mode,
        save_dir=args.save_dir,
        save_file=args.save_file,
    )
    results = evaluator.evaluate()
    print(results)


def test_performance(args):
    """Run inference performance profiling and print results.

    Measures inference latency and GPU memory usage at multiple sequence lengths
    to characterize the computational cost of the context-extended model.

    Args:
        args: Parsed CLI arguments containing model, device, length range, and
            output settings.

    Returns:
        None
    """
    model, config = load_model(args)
    tokenizer = load_tokenizer(args)
    args.save_file = args.save_file or generate_save_filename(args.model_name, config)
    evaluator = PerformanceEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        min_length=args.min_length,
        max_length=args.max_length,
        length_step=args.length_step,
        save_dir=args.save_dir,
        save_file=args.save_file,
    )
    results = evaluator.evaluate()
    print("\nTest results overview:")
    for i, length in enumerate(results["lengths"]):
        print(
            f"Input length: {length}, "
            f"Runtime: {results['runtimes'][i]:.4f}s, "
            f"Memory change: {results['memory_usages'][i]:.4f}GB"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run different evaluation tests")
    subparsers = parser.add_subparsers(dest="test", help="Type of test to run")

    # ── Perplexity ────────────────────────────────────────────────────── #
    parser_perplexity = subparsers.add_parser("perplexity", help="Run perplexity test")
    parser_perplexity = add_args_model(parser_perplexity)
    parser_perplexity = add_args_perplexity(parser_perplexity)

    # ── Passkey ───────────────────────────────────────────────────────── #
    parser_passkey = subparsers.add_parser("passkey", help="Run passkey test")
    parser_passkey = add_args_model(parser_passkey)
    parser_passkey = add_args_passkey(parser_passkey)

    # ── Quality ───────────────────────────────────────────────────────── #
    parser_quality = subparsers.add_parser("quality", help="Run quality test")
    parser_quality = add_args_model(parser_quality)
    parser_quality = add_args_quality(parser_quality)

    # ── Performance ───────────────────────────────────────────────────── #
    parser_performance = subparsers.add_parser(
        "performance",
        help="Run performance test",
    )
    parser_performance = add_args_model(parser_performance)
    parser_performance = add_args_performance(parser_performance)

    args = parser.parse_args()

    if args.test == "perplexity":
        test_perplexity(args)
    elif args.test == "passkey":
        test_passkey(args)
    elif args.test == "quality":
        test_quality(args)
    elif args.test == "performance":
        test_performance(args)
    else:
        print(f"Unknown test type: {args.test}")
        print("Use --help for usage information")
        parser.print_help()
        exit(1)
