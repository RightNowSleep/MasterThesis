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
    Performance evaluator for measuring model runtime and GPU memory usage at different input lengths.

    This evaluator tests model performance across a range of input lengths, recording
    inference time and memory consumption for each length. Results are saved to JSON
    files for further analysis.

    Attributes:
        model: Language model to evaluate
        tokenizer: Tokenizer for processing text
        device: Compute device (cuda/cpu)
        min_length: Minimum input length (token count)
        max_length: Maximum input length (token count)
        length_step: Input length step size
        length_list: List of lengths to test
        model_memory: GPU memory occupied by model weights (GB)
        save_dir: Directory to save results
        save_file: Filename to save results
    """

    def __init__(
        self,
        model,
        tokenizer,
        device=None,
        min_length=256,
        max_length=8192,
        length_step=256,
        save_dir="results/performance",
        save_file=None,
    ):
        """
        Initialize the performance evaluator.

        Args:
            model: Language model to evaluate
            tokenizer: Tokenizer for processing text
            device (str, optional): Compute device, options are "gpu", "cpu", "cuda";
                default is None (auto-select)
            min_length (int, optional): Minimum input length in tokens, default is 256
            max_length (int, optional): Maximum input length in tokens, default is 8192
            length_step (int, optional): Input length step size in tokens, default is 256
            save_dir (str, optional): Directory to save results, default is "results/performance"
            save_file (str, optional): Filename to save results, default is None (auto-generate)
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
        self.length_step = length_step
        self.length_list = self._generate_length_list()

        # Record model weight memory usage upfront, so it can be included in
        # each per-length memory report (activation delta alone is misleading)
        self.model_memory = (
            sum(p.element_size() * p.nelement() for p in self.model.parameters())
            / 1024**3
        )

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_file = (
            save_file if save_file else f"{time.strftime('%Y%m%d-%H%M%S')}.json"
        )

    def _generate_length_list(self):
        """
        Generate a list of test lengths.

        Creates a list of input lengths from min_length to max_length (inclusive),
        incremented by step_length each time.

        Returns:
            list: List of input lengths to test, e.g., [256, 512, 768, ...]
        """
        if self.length_step is not None:
            length_list = [
                x for x in range(self.min_length, self.max_length + 1, self.length_step)
            ]
        else:
            length_list = []
            current_length = self.min_length
            while current_length <= self.max_length:
                length_list.append(current_length)
                current_length += self.length_step
        return length_list

    def _get_gpu_memory(self):
        """
        Get total GPU memory usage across all devices.

        Returns:
            float: Total GPU memory allocated in GB across all devices,
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
            length (int): Target token length

        Returns:
            list: List of token IDs with exactly the specified length
        """
        sample_text = "The quick brown fox jumps over the lazy dog. "
        tokens = self.tokenizer.encode(sample_text, add_special_tokens=False)

        repeated_tokens = []
        while len(repeated_tokens) < length:
            repeated_tokens.extend(tokens)

        repeated_tokens = repeated_tokens[:length]
        return repeated_tokens

    def _synchronize_all_devices(self):
        """Synchronize all available CUDA devices."""
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)

    def _reset_peak_memory_stats_all_devices(self):
        """Reset peak memory statistics on all available CUDA devices."""
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)

    def _get_peak_gpu_memory(self):
        """
        Get peak GPU memory across all devices since the last reset.

        Returns:
            float: Sum of peak memory allocated (GB) across all CUDA devices,
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
        initialization overhead, making its runtime appear artificially high.
        """
        print("Warming up...")
        with torch.no_grad():
            warmup_tokens = self._generate_sample_text(self.min_length)
            warmup_ids = torch.tensor([warmup_tokens]).to(self.device)
            self.model(warmup_ids)
            del warmup_ids
        if torch.cuda.is_available():
            self._synchronize_all_devices()
            torch.cuda.empty_cache()
        gc.collect()

    @torch.no_grad()
    def test_performance_at_length(self, input_length):
        """
        Test performance at a specified input length.

        Measures inference runtime and total GPU memory usage (model weights +
        peak activation) for a single forward pass with the given input length.

        Args:
            input_length (int): Input sequence length in tokens

        Returns:
            tuple: (runtime, memory_usage)
                - runtime (float): Inference time in seconds
                - memory_usage (float): Total GPU memory in GB
                  (model weights + peak activation during forward pass)
        """
        # Clear cache before each test for a clean baseline
        torch.cuda.empty_cache()
        gc.collect()

        input_tokens = self._generate_sample_text(input_length)
        input_ids = torch.tensor([input_tokens]).to(self.device)

        # Record baseline memory across all devices before the forward pass.
        # This captures any non-weight allocations already present (e.g. CUDA
        # context, optimizer states) so they are not mis-attributed to activations.
        initial_memory = self._get_gpu_memory()

        # Reset peak trackers on every device so max_memory_allocated reflects
        # only the allocations that occur during this forward pass.
        self._reset_peak_memory_stats_all_devices()

        # Synchronize all devices before starting the timer to ensure no
        # leftover async GPU work leaks into the measurement.
        self._synchronize_all_devices()
        start_time = time.time()
        outputs = self.model(input_ids)
        # Synchronize all devices again so the clock stops only after every
        # GPU has truly finished (GPU ops are asynchronous by default).
        self._synchronize_all_devices()
        end_time = time.time()

        runtime = end_time - start_time

        # Peak memory across all devices since the reset
        peak_memory = self._get_peak_gpu_memory()
        # Activation memory = peak during forward - baseline before forward.
        # Adding model_memory separately avoids double-counting: peak_memory
        # already includes the model weights that were present at reset time,
        # so we subtract initial_memory (which contains those weights) and
        # then add back the canonical model_memory figure computed from
        # parameter sizes in __init__.
        activation_memory = peak_memory - initial_memory
        memory_usage = self.model_memory + activation_memory

        # Clean up
        del input_ids, outputs
        torch.cuda.empty_cache()

        return runtime, memory_usage

    def evaluate(self):
        """
        Run performance tests at all configured lengths.

        Iterates through all lengths in length_list, measures runtime and
        memory usage for each, and saves results to a JSON file.

        Returns:
            dict: Evaluation results containing:
                - lengths (list): List of input sequence lengths
                - runtimes (list): List of inference times in seconds
                - memory_usages (list): List of total GPU memory in GB
        """
        self._warmup()

        lengths = []
        runtimes = []
        memory_usages = []

        print(
            f"Starting performance test, {len(self.length_list)} length points in total"
        )
        pbar = tqdm(total=len(self.length_list), desc="Performance Testing")

        for length in self.length_list:
            runtime, memory_usage = self.test_performance_at_length(length)

            lengths.append(length)
            runtimes.append(runtime)
            memory_usages.append(memory_usage)

            pbar.set_postfix(
                {
                    "Length": length,
                    "Runtime": f"{runtime:.4f}s",
                    "Memory": f"{memory_usage:.4f}GB",
                }
            )
            pbar.update(1)

        pbar.close()

        save_path = os.path.join(self.save_dir, self.save_file)
        with open(save_path, "w") as f:
            json.dump(
                {
                    "lengths": lengths,
                    "runtimes": runtimes,
                    "memory_usages": memory_usages,
                },
                f,
                indent=4,
            )

        print(f"Performance test completed, results saved to: {save_path}")
        return {
            "lengths": lengths,
            "runtimes": runtimes,
            "memory_usages": memory_usages,
        }


def generate_save_filename(args):
    """
    Generate filename based on model and RoPE configuration.

    Examples:
        --rope-type none                              → llama-7b_none.json
        --rope-type linear --rope-dynamic             → llama-7b_linear_dynamic.json
        --rope-type linear --rope-factor 4.0          → llama-7b_linear_factor4_0.json
        --rope-type ntk --rope-factor 2.5             → llama-7b_ntk_factor2_5.json
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


def add_args_performance(parser):
    r"""
    Add performance testing related arguments to argument parser.

    Args:
        parser: ArgumentParser object to add arguments to

    Returns:
        ArgumentParser: Parser with added arguments
    """
    parser.add_argument(
        "--length-step",
        type=int,
        default=None,
        help="Input length step size in tokens",
    )
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

    args.save_file = args.save_file or generate_save_filename(args)

    # Create performance evaluator and run tests
    tester = PerformanceEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        min_length=args.min_length,
        max_length=args.max_length,
        length_step=args.length_step,
        save_dir=args.save_dir,
        save_file=args.save_file,
    )

    results = tester.evaluate()
    print("\nTest results overview:")
    for i, length in enumerate(results["lengths"]):
        print(
            f"Input length: {length}, "
            f"Runtime: {results['runtimes'][i]:.4f}s, "
            f"Memory change: {results['memory_usages'][i]:.4f}GB"
        )
