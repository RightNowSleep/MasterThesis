"""
eval/performance.py
---------------------
Performance evaluator for measuring model inference runtime and GPU memory usage.

Tests a language model from min_length to max_length (incrementing by 1 token each step),
recording:
    - Total inference time (seconds) across all forward passes.
    - GPU peak memory difference (GB) between start and end of evaluation.

Results are saved as JSON for further analysis and comparison across different
RoPE configurations or model variants.

Usage:
    python eval/performance.py --model-name huggyllama/llama-7b --max-length 8192
"""

import gc
import torch
import time
import os
import json
import argparse
from tqdm import tqdm
from models.model_loader import load_model, load_tokenizer, add_args_model


class PerformanceEvaluator:
    """
    Performance evaluator for measuring model runtime and GPU memory usage
    with autoregressive generation.

    This evaluator simulates autoregressive generation from min_length to
    max_length (+1 token each step). It respects the model's use_cache config:
    if KV Cache is enabled, it uses past_key_values for efficient decoding;
    if KV Cache is disabled, it passes the full sequence each step.
    It records total inference time and GPU peak memory difference.
    Results are saved to JSON files for further analysis.

    Attributes:
        model: Language model to evaluate.
        tokenizer: Tokenizer for processing text.
        device: Compute device (cuda/cpu).
        min_length: Minimum input length (token count).
        max_length: Maximum input length (token count).
        save_dir: Directory to save results.
        save_file: Filename to save results.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device=None,
        min_length=256,
        max_length=8192,
        save_dir="results/performance",
        save_file=None,
    ):
        """
        Initialize the performance evaluator.

        Args:
            model: Language model to evaluate.
            tokenizer: Tokenizer for processing text.
            device: Compute device, options are "gpu", "cpu", "cuda";
                default is None (auto-select).
            min_length: Minimum input length in tokens, default is 256.
            max_length: Maximum input length in tokens, default is 8192.
            save_dir: Directory to save results, default is "results/performance".
            save_file: Filename to save results, default is None (auto-generate).
        """
        self.model = model.eval()
        self.tokenizer = tokenizer

        if device is not None:
            assert device in [
                "gpu",
                "cpu",
                "cuda",
            ], "device should be either gpu or cpu."
            if device == "gpu":
                self.device = "cuda"
            else:
                self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.min_length = min_length
        self.max_length = max_length

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_file = (
            save_file if save_file else f"{time.strftime('%Y%m%d-%H%M%S')}.json"
        )

    def _get_gpu_memory(self):
        """
        Get total GPU memory usage across all devices.

        Returns:
            Total GPU memory allocated in GB across all devices,
            or 0 if CUDA is not available.
        """
        if not torch.cuda.is_available():
            return 0

        total_memory = 0.0
        num_devices = torch.cuda.device_count()
        for i in range(num_devices):
            total_memory += torch.cuda.memory_allocated(i) / 1024**3
        return total_memory

    def _generate_sample_text(self, length):
        """
        Generate sample text of specified token length.

        Creates a repeated pattern of sample text tokens to reach the
        desired input length.

        Args:
            length: Target token length.

        Returns:
            List of token IDs with exactly the specified length.
        """
        sample_text = "The quick brown fox jumps over the lazy dog. "
        tokens = self.tokenizer.encode(sample_text, add_special_tokens=False)

        repeated_tokens = []
        while len(repeated_tokens) < length:
            repeated_tokens.extend(tokens)

        repeated_tokens = repeated_tokens[:length]
        return repeated_tokens

    def _synchronize_all_devices(self):
        """
        Synchronize all available CUDA devices.

        Blocks the calling thread until all pending GPU operations on every
        visible CUDA device have completed. Used to ensure accurate timing
        measurements.

        Returns:
            None.
        """
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)

    def _reset_peak_memory_stats_all_devices(self):
        """
        Reset peak memory statistics on all available CUDA devices.

        After calling this, max_memory_allocated() will only reflect
        allocations that occur after the reset. Used before each forward
        pass to isolate per-run activation memory.

        Returns:
            None.
        """
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)

    def _get_peak_gpu_memory(self):
        """
        Get peak GPU memory across all devices since the last reset.

        Returns:
            Sum of peak memory allocated (GB) across all CUDA devices,
            or 0 if CUDA is not available.
        """
        if not torch.cuda.is_available():
            return 0.0
        return sum(
            torch.cuda.max_memory_allocated(i) / 1024**3
            for i in range(torch.cuda.device_count())
        )

    def _warmup(self):
        """
        Run a warmup forward pass to trigger CUDA kernel compilation and loading.

        Without warmup, the first real test point would include one-time CUDA
        initialisation overhead, making its runtime appear artificially high.
        Uses the minimum configured length for the warmup sequence.

        Returns:
            None.
        """
        print("Warming up...")
        with torch.no_grad():
            warmup_tokens = self._generate_sample_text(self.min_length)
            warmup_ids = torch.tensor([warmup_tokens]).to(self.device)
            self.model(warmup_ids, use_cache=self.model.config.use_cache)
            del warmup_ids
        if torch.cuda.is_available():
            self._synchronize_all_devices()
            torch.cuda.empty_cache()
        gc.collect()

    @torch.no_grad()
    def evaluate(self):
        """
        Run performance test simulating autoregressive generation.

        From min_length to max_length (+1 token each step), respecting the
        model's use_cache configuration:
        - If use_cache=True: prefill min_length tokens, then decode one token
          at a time with past_key_values (KV Cache).
        - If use_cache=False: pass the full sequence each step (no KV Cache).

        Measures total inference runtime and GPU peak memory difference between
        start and end of the evaluation. Results are saved to a JSON file.

        Returns:
            Evaluation results containing:
                min_length: Starting input length.
                max_length: Ending input length.
                use_cache: Whether KV Cache was used.
                total_runtime_seconds: Total inference time in seconds (2 decimal places).
                peak_memory_start_gb: GPU allocated memory at start in GB (2 decimal places).
                peak_memory_end_gb: GPU peak memory at end in GB (2 decimal places).
                memory_diff_gb: Peak memory difference in GB (2 decimal places).
        """
        self._warmup()

        use_cache = self.model.config.use_cache

        self._reset_peak_memory_stats_all_devices()
        start_memory = self._get_gpu_memory()

        sample_text = "The quick brown fox jumps over the lazy dog. "
        base_tokens = self.tokenizer.encode(sample_text, add_special_tokens=False)
        all_tokens = []
        while len(all_tokens) < self.max_length:
            all_tokens.extend(base_tokens)
        all_tokens = all_tokens[: self.max_length]

        total_steps = self.max_length - self.min_length

        if use_cache:
            print(
                f"Starting performance test with KV Cache, "
                f"prefill {self.min_length} tokens then {total_steps} autoregressive steps"
            )
        else:
            print(
                f"Starting performance test without KV Cache, "
                f"{total_steps + 1} full-sequence forward passes"
            )

        pbar = tqdm(total=total_steps + 1, desc="Performance Testing")

        self._synchronize_all_devices()
        overall_start_time = time.time()

        if use_cache:
            prefill_ids = torch.tensor([all_tokens[: self.min_length]]).to(self.device)
            outputs = self.model(prefill_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            del prefill_ids, outputs

            pbar.set_postfix({"Length": self.min_length, "Phase": "prefill"})
            pbar.update(1)

            for i in range(self.min_length, self.max_length):
                new_token_id = torch.tensor([[all_tokens[i]]]).to(self.device)
                outputs = self.model(
                    new_token_id,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                del new_token_id, outputs

                pbar.set_postfix(
                    {
                        "Length": i + 1,
                        "Elapsed": f"{time.time() - overall_start_time:.2f}s",
                    }
                )
                pbar.update(1)

            del past_key_values
        else:
            for length in range(self.min_length, self.max_length + 1):
                input_ids = torch.tensor([all_tokens[:length]]).to(self.device)
                outputs = self.model(input_ids, use_cache=False)
                del input_ids, outputs

                if (length - self.min_length) % 1024 == 0 and length > self.min_length:
                    torch.cuda.empty_cache()
                    gc.collect()

                pbar.set_postfix(
                    {
                        "Length": length,
                        "Elapsed": f"{time.time() - overall_start_time:.2f}s",
                    }
                )
                pbar.update(1)

        pbar.close()

        self._synchronize_all_devices()
        overall_end_time = time.time()

        end_peak_memory = self._get_peak_gpu_memory()
        total_runtime = round(overall_end_time - overall_start_time, 2)
        memory_diff = round(end_peak_memory - start_memory, 2)
        start_memory = round(start_memory, 2)
        end_peak_memory = round(end_peak_memory, 2)

        torch.cuda.empty_cache()
        gc.collect()

        save_path = os.path.join(self.save_dir, self.save_file)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "min_length": self.min_length,
                    "max_length": self.max_length,
                    "use_cache": use_cache,
                    "total_runtime_seconds": total_runtime,
                    "peak_memory_start_gb": start_memory,
                    "peak_memory_end_gb": end_peak_memory,
                    "memory_diff_gb": memory_diff,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"Performance test completed, results saved to: {save_path}")
        return {
            "min_length": self.min_length,
            "max_length": self.max_length,
            "use_cache": use_cache,
            "total_runtime_seconds": total_runtime,
            "peak_memory_start_gb": start_memory,
            "peak_memory_end_gb": end_peak_memory,
            "memory_diff_gb": memory_diff,
        }


def generate_save_filename(model_name, config):
    """
    Generate filename based on model and RoPE configuration.

    Args:
        model_name: The name of the model.
        config: The configuration object.

    Returns:
        The filename.

    Examples:
        --rope-type none                              → llama-7b_none.json
        --rope-type linear --rope-dynamic             → llama-7b_linear_dynamic.json
        --rope-type linear --rope-factor 4.0          → llama-7b_linear_factor4_0.json
        --rope-type ntk --rope-factor 2.5             → llama-7b_ntk_factor2_5.json
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


def add_args_performance(parser):
    r"""
    Add performance testing related arguments to argument parser.

    Args:
        parser: ArgumentParser object to add arguments to.

    Returns:
        Parser with added arguments.
    """
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/performance",
        help="Directory to save performance test results",
    )
    parser.add_argument(
        "--save-file",
        type=str,
        default=None,
        help="Filename to save performance test results",
    )
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args_model(parser)
    parser = add_args_performance(parser)
    args = parser.parse_args()

    model, config = load_model(args)
    tokenizer = load_tokenizer(args)

    args.save_file = args.save_file or generate_save_filename(args.model_name, config)

    # Create performance evaluator and run tests
    tester = PerformanceEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        min_length=args.min_length,
        max_length=args.max_length,
        save_dir=args.save_dir,
        save_file=args.save_file,
    )

    results = tester.evaluate()
    print("\nTest results overview:")
    print(
        f"Length range: {results['min_length']} - {results['max_length']}, "
        f"Total runtime: {results['total_runtime_seconds']}s, "
        f"Memory diff: {results['memory_diff_gb']}GB"
    )
