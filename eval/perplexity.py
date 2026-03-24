import gc
import torch
from datasets import load_dataset
from tqdm import tqdm
import argparse
import time
import os
import json
from models.model_loader import load_model, load_tokenizer, add_args_model


class PerplexityEvaluator:
    r"""
    Perplexity evaluator for computing language model perplexity on a given dataset.

    Perplexity is an important metric for evaluating language model performance; lower values indicate better predictive capability.
    This evaluator uses a sliding window approach to handle long texts, avoiding exceeding the model's maximum context length limit.

    Supports evaluating models at multiple different context lengths, e.g., 256, 512, 1024, 2048, 4096, 8192, etc.

    Attributes:
        model: Language model for computing perplexity
        tokenizer: Tokenizer for processing text
        dataset_path: Dataset path or name, default is "emozilla/proofpile-test-tokenized"
        split: Dataset split, default is "test"
        limit: Number of samples to evaluate, default is 100
        device: Compute device (cuda/cpu)
        add_start_token: Whether to add BOS token at the beginning of the sequence
        max_length: Maximum sequence length
        min_length: Minimum sequence length
        length_step: Length step size; if None, uses exponential growth (multiply by 2 each time)
        sliding_window: Sliding window size
        truncate: Whether to truncate sequences exceeding max_length
        aggressive_memory: Whether to enable aggressive memory management
        max_tokenized_len: Maximum token length (considering BOS token)
        dataset: Loaded dataset
        length_list: List of lengths to evaluate
        save_dir: Directory to save results, default is "results/perplexity"
        save_file: Filename to save results, default is None
    """

    def __init__(
        self,
        model,
        tokenizer,
        dataset="emozilla/proofpile-test-tokenized",
        split="test",
        limit=None,
        device=None,
        add_start_token=True,
        max_length=32768,
        min_length=2048,
        length_step=None,
        sliding_window=256,
        truncate=True,
        aggressive_memory=True,
        save_dir="results/perplexity",
        save_file=None,
    ):
        r"""
        Initialize the perplexity evaluator.

        Args:
            model: Language model (required), pretrained model for computing perplexity
            tokenizer: Tokenizer (required), tokenizer for processing text
            dataset (str, optional): Dataset path or name, default is "emozilla/proofpile-test-tokenized"
            split (str, optional): Dataset split, default is "test"
            limit (int, optional): Number of samples to evaluate, default is 100
            device (str, optional): Compute device, options are "gpu", "cpu", "cuda", default is None (auto-select)
            add_start_token (bool, optional): Whether to add BOS token at sequence start, default is True
                If set to True, the model must have a BOS token
            max_length (int, optional): Maximum sequence length, default is 8192
                Maximum context length for evaluation
            min_length (int, optional): Minimum sequence length, default is 256
                Minimum context length for evaluation
            length_step (int, optional): Length step size, default is None
                If set, grows at fixed step size (e.g., 256, 512, 768, 1024...)
                If None, uses exponential growth (e.g., 256, 512, 1024, 2048...)
            sliding_window (int, optional): Sliding window size, default is 256
                Used for processing long texts; window size affects computational efficiency and memory usage
            truncate (bool, optional): Whether to truncate sequences exceeding max_length, default is False
                If set to True, parts exceeding max_length will be discarded
            aggressive_memory (bool, optional): Whether to enable aggressive memory management, default is False
                If set to True, memory is cleared after each window processing, reducing memory usage but potentially slowing down
            save_dir (str, optional): Directory to save results, default is "results/perplexity"
            save_file (str, optional): Filename to save results, default is None

        Raises:
            AssertionError: Raised when add_start_token=True but model has no BOS token
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
        self.sliding_window = sliding_window
        self.truncate = truncate
        self.aggressive_memory = aggressive_memory

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

        if self.add_start_token:
            assert (
                self.tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            self.max_tokenized_len = self.max_length - 1
        else:
            self.max_tokenized_len = self.max_length

        self.dataset = load_dataset(self.dataset_path, split=self.split)
        print(f"Dataset size: {len(self.dataset)}")
        if self.limit is not None:
            self.dataset = self.dataset[: self.limit]
        self.length_list = self._generate_length_list()

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_file = (
            save_file if save_file else f"{time.strftime('%Y%m%d-%H%M%S')}.json"
        )

    def _generate_length_list(self):
        r"""
        Generate a list of lengths for evaluating the model at different context lengths.

        If length_step is not None, grows at fixed step size (e.g., 256, 512, 768, 1024...)
        If length_step is None, uses exponential growth (e.g., 256, 512, 1024, 2048...)

        Returns:
            list: List of lengths, e.g., [256, 512, 1024, 2048, 4096, 8192]
        """
        if self.length_step is not None:
            length_list = [
                x for x in range(self.min_length, self.max_length + 1, self.length_step)
            ]
        else:
            length_list = [self.min_length]
            while length_list[-1] < self.max_length:
                point = length_list[-1] * 2
                if point <= self.max_length:
                    length_list.append(point)
                else:
                    break
        return length_list

    @torch.no_grad()
    def _compute_perplexity_at_length(self, max_length):
        r"""
        Compute perplexity at a specified length.

        Args:
            max_length (int): Maximum sequence length

        Returns:
            float: Average perplexity at the specified length
        """
        encoded_texts = self.dataset["input_ids"]
        attn_masks = self.dataset["attention_mask"]

        if self.add_start_token:
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        if self.truncate:
            encoded_texts = [x[0:max_tokenized_len] for x in encoded_texts]
            attn_masks = [x[0:max_tokenized_len] for x in attn_masks]
            sliding_window = max_tokenized_len
        else:
            sliding_window = self.sliding_window

        pbar = tqdm(total=len(encoded_texts))
        # FIX 1: Use weighted accumulation, record sum of NLL*trg_len and total token count separately
        nll_sum = torch.tensor(0.0)
        total_tokens = 0

        for encoding_index in range(0, len(encoded_texts)):

            labels = torch.tensor(encoded_texts[encoding_index : encoding_index + 1])
            seq_len = labels.size(1)

            # FIX 2: Synchronously build attention_mask tensor, pass to model later
            attn_mask = torch.tensor(attn_masks[encoding_index : encoding_index + 1])

            prev_end_loc = 0
            for begin_loc in range(0, seq_len, sliding_window):

                end_loc = min(begin_loc + max_tokenized_len, seq_len)
                trg_len = end_loc - prev_end_loc
                input_ids = labels[:, begin_loc:end_loc].to(self.device)
                attn_mask_slice = attn_mask[:, begin_loc:end_loc].to(self.device)

                if self.add_start_token:
                    bos_tokens_tensor = torch.tensor(
                        [[self.tokenizer.bos_token_id]] * input_ids.size(dim=0)
                    ).to(self.device)
                    input_ids = torch.cat([bos_tokens_tensor, input_ids], dim=1)
                    # BOS token attention mask is always 1
                    bos_attn = torch.ones(
                        (attn_mask_slice.size(0), 1),
                        dtype=attn_mask_slice.dtype,
                    ).to(self.device)
                    attn_mask_slice = torch.cat([bos_attn, attn_mask_slice], dim=1)

                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                # FIX 2: Pass attention_mask to model
                outputs = self.model(
                    input_ids,
                    attention_mask=attn_mask_slice,
                    labels=target_ids,
                )
                neg_log_likelihood = outputs.loss.clone()

                # FIX 1: Accumulate weighted by window's valid token count
                nll_sum += neg_log_likelihood.cpu() * trg_len
                total_tokens += trg_len

                ppl = float(torch.exp(nll_sum / total_tokens).float())
                pbar.set_postfix(ppl=ppl)

                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break

            if self.aggressive_memory:
                del labels, attn_mask
                gc.collect()
                torch.cuda.empty_cache()

            pbar.update(1)
        pbar.close()

        ppl = float(torch.exp(nll_sum / total_tokens).float())
        return ppl

    def evaluate(self):
        r"""
        Compute average perplexity on the dataset at different context lengths.

        Uses sliding window method to compute perplexity, as validated in arXiv 2306.15595.
        Sliding windows can handle long texts that exceed the model's maximum context length.

        Computation process:
        1. Generate a list of lengths (e.g., 256, 512, 1024, 2048, 4096, 8192)
        2. For each length, split texts in the dataset into multiple sliding windows
        3. Compute negative log likelihood (NLL) for each window, weighted by token count
        4. Accumulate weighted NLL across all windows and divide by total token count
        5. Compute perplexity: exp(weighted_mean(NLL))
        6. Return perplexity results at all lengths

        Returns:
            dict: Dictionary containing perplexity at all lengths
                - lengths (list): List of evaluated lengths, e.g., [256, 512, 1024, 2048, 4096, 8192]
                - perplexities (list): List of perplexities at corresponding lengths

        Example:
            >>> evaluator = PerplexityEvaluator(model, tokenizer, max_length=8192)
            >>> result = evaluator.evaluate()
            >>> print(result["lengths"])
            [256, 512, 1024, 2048, 4096, 8192]
            >>> print(result["perplexities"])
            [12.34, 11.23, 10.45, 9.87, 9.32, 8.95]
        """
        perplexities = []

        pbar = tqdm(
            total=len(self.length_list),
            desc="Perplexity Evaluation",
            leave=False,
        )
        for max_length in self.length_list:
            torch.cuda.empty_cache()
            pbar.set_postfix(max_length=max_length)
            ppl = self._compute_perplexity_at_length(max_length)
            perplexities.append(ppl)
            pbar.update(1)

        pbar.close()

        save_path = os.path.join(self.save_dir, self.save_file)
        with open(save_path, "w") as f:
            json.dump(
                {
                    "lengths": self.length_list,
                    "perplexities": perplexities,
                },
                f,
                indent=2,
            )

        return {
            "lengths": self.length_list,
            "perplexities": perplexities,
        }


def generate_save_filename(model_name, config):
    """
    Generate filename based on model and RoPE configuration.

    Args:
        model_name: The name of the model.
        config: The configuration object.

    Returns:
        str: The filename.

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


def add_args_perplexity(parser):
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="emozilla/proofpile-test-tokenized",
        help="Dataset name",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--limit", type=int, default=100, help="Dataset limit")
    parser.add_argument(
        "--add-start-token",
        type=bool,
        default=True,
        help="Whether to add BOS token",
    )
    parser.add_argument(
        "--length-step",
        type=int,
        default=None,
        help="Length step size",
    )
    parser.add_argument(
        "--sliding-window",
        type=int,
        default=256,
        help="Sliding window size",
    )
    parser.add_argument(
        "--truncate",
        type=bool,
        default=True,
        help="Whether to truncate",
    )
    parser.add_argument(
        "--aggressive-memory",
        type=bool,
        default=True,
        help="Aggressive memory management",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/perplexity",
        help="Save directory",
    )
    parser.add_argument(
        "--save-file",
        type=str,
        default=None,
        help="Save filename",
    )
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args_model(parser)
    parser = add_args_perplexity(parser)
    args = parser.parse_args()
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
