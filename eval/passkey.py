import re
import torch
import random
from abc import ABC, abstractmethod
from transformers import pipeline
from datasets import load_dataset
from tqdm import tqdm, trange
import gc
import time
import os
import argparse
import json

from models.model_loader import load_model, load_tokenizer, add_args_model

random.seed(42)


def get_order_suffix(i: int) -> str:
    """
    Generate ordinal suffix

    Args:
        i (int): Number

    Returns:
        str: Ordinal suffix string, e.g., 1->"1st", 2->"2nd", 3->"3rd", 4->"4th"
    """
    if 11 <= i % 100 <= 13:
        return f"{i}th"
    elif i % 10 == 1:
        return f"{i}st"
    elif i % 10 == 2:
        return f"{i}nd"
    elif i % 10 == 3:
        return f"{i}rd"
    else:
        return f"{i}th"


class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders

    Defines a unified interface for data loaders, subclasses need to implement the generate_prompt method
    """

    def __init__(self, tokenizer, length: int):
        """
        Initialize data loader

        Args:
            tokenizer: Tokenizer object
            length (int): Target token count
        """
        self.tokenizer = tokenizer
        self.length = length
        self.task_description = (
            "There is an important info hidden inside a lot of irrelevant text. "
            "Find it and memorize them. I will quiz you about the important information there.\n"
        )

    def set_length(self, length: int):
        """
        Set target token count

        Args:
            length (int): Target token count
        """
        self.length = length

    @abstractmethod
    def generate_prompt(self, num_keys: int = 1) -> tuple:
        """
        Generate test prompt (abstract method)

        Args:
            num_keys (int): Number of pass keys, default is 1

        Returns:
            tuple: (prompt_text, pass_keys, target_key)
                - prompt_text (str): Generated test prompt text
                - pass_keys (list): List of all pass keys
                - target_key (int): Target pass key for the model to answer
        """
        pass


class SyntheticDataLoader(BaseDataLoader):
    """Generates synthetic corpus using repeated garbage text for passkey testing"""

    def __init__(self, tokenizer, length: int):
        """
        Initialize synthetic data loader

        Args:
            tokenizer: Tokenizer object
            length (int): Target token count
        """
        super().__init__(tokenizer, length)
        self.garbage = "The grass is green. The sky is blue. The sun is three. Here we go. There and back again."
        self.garbage_tokens = self.tokenizer.encode(self.garbage)

    def generate_prompt(self, num_keys: int = 1) -> tuple:
        """
        Generate synthetic corpus test prompt

        Args:
            num_keys (int): Number of pass keys, default is 1

        Returns:
            tuple: (prompt_text, pass_keys, target_key)
                - prompt_text (str): Generated test prompt text
                - pass_keys (list): List of all pass keys
                - target_key (int): Target pass key for the model to answer
        """
        pass_keys = [random.randint(1, 50000) for _ in range(num_keys)]

        information_lines = [
            f"The {get_order_suffix(i+1)} pass key is {key}. "
            f"Remember it. {key} is the {get_order_suffix(i+1)} pass key."
            for i, key in enumerate(pass_keys)
        ]

        target_idx = random.randint(0, num_keys - 1)
        target_key = pass_keys[target_idx]

        final_question = (
            f"\nWhat is the {get_order_suffix(target_idx + 1)} pass key? "
            f"The {get_order_suffix(target_idx + 1)} pass key is"
        )

        task_description_tokens = self.tokenizer.encode(self.task_description)
        final_question_tokens = self.tokenizer.encode(final_question)
        fixed_tokens = task_description_tokens + final_question_tokens

        garbage_tokens_needed = self.length - len(fixed_tokens)

        if garbage_tokens_needed > 0:
            num_repeats = (garbage_tokens_needed // len(self.garbage_tokens)) + 1
            repeated_garbage = self.garbage * num_repeats
            repeated_garbage_tokens = self.tokenizer.encode(repeated_garbage)
            garbage_tokens = repeated_garbage_tokens[:garbage_tokens_needed]
        else:
            garbage_tokens = []

        information_tokens = [
            self.tokenizer.encode(info_line) for info_line in information_lines
        ]

        lines = [self.task_description]
        prev_pos = 0

        for i, info_line in enumerate(information_lines):
            info_token_count = len(information_tokens[i])

            if len(garbage_tokens) > 0:
                remaining_info_tokens = sum(len(t) for t in information_tokens[i:])
                max_insert_pos = len(garbage_tokens) - remaining_info_tokens

                if max_insert_pos > prev_pos:
                    insert_pos = random.randint(prev_pos, max_insert_pos)
                else:
                    insert_pos = prev_pos

                garbage_segment = self.tokenizer.decode(
                    garbage_tokens[prev_pos:insert_pos]
                )
                lines.append(garbage_segment)
                lines.append(info_line)
                garbage_tokens = garbage_tokens[insert_pos + info_token_count :]
                prev_pos = 0
            else:
                lines.append(info_line)

        if len(garbage_tokens) > 0:
            final_garbage = self.tokenizer.decode(garbage_tokens)
            lines.append(final_garbage)

        lines.append(final_question)
        prompt_text = "".join(lines)

        return prompt_text, pass_keys, target_key


class RealDataLoader(BaseDataLoader):
    """Loads text from real datasets for passkey testing"""

    def __init__(
        self,
        dataset_name: str = "konwoo/RedPajama-Data-1T-Sample-subset1000",
        split: str = "train",
        tokenizer=None,
        length: int = 8192,
    ):
        """
        Initialize real data loader

        Args:
            dataset_name (str): Dataset name, e.g., "togethercomputer/RedPajama-Data-1T-Sample"
            split (str): Dataset split, e.g., "train", "test"
            tokenizer: Tokenizer object
            length (int): Total text length (token count)
        """
        super().__init__(tokenizer, length)
        self.dataset_name = dataset_name
        self.split = split
        self.docs = self._construct_junk()

    def set_length(self, length: int):
        """
        Set target token count and rebuild junk text

        Args:
            length (int): Target token count
        """
        self.length = length
        self.docs = self._construct_junk()

    def _construct_junk(self):
        """
        Construct junk text

        Randomly sample text from real dataset until reaching specified token count

        Returns:
            list: List of pre-tokenized document segments
        """
        data = load_dataset(self.dataset_name)[self.split]
        token_count = 0
        docs = []

        while token_count < self.length:
            sample = random.choice(data)["text"]
            toks = self.tokenizer(sample, return_offsets_mapping=True)
            offsets = [(i, j) for i, j in toks["offset_mapping"] if i < j]
            num_tok_to_add = min(self.length - token_count, len(offsets))
            pretokenized = [sample[i:j] for i, j in offsets[:num_tok_to_add]]
            docs.extend(pretokenized)
            token_count += num_tok_to_add

        return docs

    def generate_prompt(self, num_keys: int = 1) -> tuple:
        """
        Generate real corpus test prompt

        Args:
            num_keys (int): Number of pass keys, default is 1

        Returns:
            tuple: (prompt_text, pass_keys, target_key)
                - prompt_text (str): Generated test prompt text
                - pass_keys (list): List of all pass keys
                - target_key (int): Target pass key for the model to answer
        """
        pass_keys = [random.randint(1, 50000) for _ in range(num_keys)]

        information_lines = [
            f"The {get_order_suffix(i+1)} pass key is {key}. "
            f"Remember it. {key} is the {get_order_suffix(i+1)} pass key."
            for i, key in enumerate(pass_keys)
        ]

        target_idx = random.randint(0, num_keys - 1)
        target_key = pass_keys[target_idx]

        final_question = (
            f"\nWhat is the {get_order_suffix(target_idx + 1)} pass key? "
            f"The {get_order_suffix(target_idx + 1)} pass key is"
        )

        task_description_tokens = self.tokenizer.encode(self.task_description)
        final_question_tokens = self.tokenizer.encode(final_question)
        fixed_tokens = task_description_tokens + final_question_tokens

        garbage_tokens_needed = self.length - len(fixed_tokens)

        if garbage_tokens_needed > 0:
            garbage_docs = self.docs[:garbage_tokens_needed]
        else:
            garbage_docs = []

        information_tokens = [
            self.tokenizer.encode(info_line) for info_line in information_lines
        ]

        lines = [self.task_description]
        prev_pos = 0

        for i, info_line in enumerate(information_lines):
            info_token_count = len(information_tokens[i])

            if len(garbage_docs) > 0:
                remaining_info_tokens = sum(len(t) for t in information_tokens[i:])
                max_insert_pos = len(garbage_docs) - remaining_info_tokens

                if max_insert_pos > prev_pos:
                    insert_pos = random.randint(prev_pos, max_insert_pos)
                else:
                    insert_pos = prev_pos

                garbage_segment = "".join(garbage_docs[prev_pos:insert_pos])
                lines.append(garbage_segment)
                lines.append(info_line)
                garbage_docs = garbage_docs[insert_pos + info_token_count :]
                prev_pos = 0
            else:
                lines.append(info_line)

        if len(garbage_docs) > 0:
            final_garbage = "".join(garbage_docs)
            lines.append(final_garbage)

        lines.append(final_question)
        prompt_text = "".join(lines)

        return prompt_text, pass_keys, target_key


class PasskeyEvaluator:
    """
    Passkey evaluator for needle-in-a-haystack retrieval testing.

    Args:
        model: Loaded model object
        tokenizer: Tokenizer object
        restrict_tokens (bool): Whether to restrict output tokens (digits only), default is True
        data_mode (str): Data mode, "synthetic" or "real"
        dataset_name (str): Dataset name, only used for real mode
        split (str): Dataset split, only used for real mode
        min_length (int): Minimum token count, default is 2048
        max_length (int): Maximum token count, default is 32768
        length_step (int): Length step size; if None, uses exponential growth (multiply by 2 each time)
        iterations (int): Number of test iterations, default is 20
        num_keys (int): Number of pass keys, default is 1
        aggressive_memory (bool): Whether to clear memory after each iteration, default is True
        save_dir (str): Save directory, default is "results/passkey"
        save_file (str): Save filename, default is None
    """

    def __init__(
        self,
        model,
        tokenizer,
        restrict_tokens: bool = True,
        data_mode: str = "synthetic",
        dataset_name: str = "konwoo/RedPajama-Data-1T-Sample-subset1000",
        split: str = "train",
        min_length: int = 2048,
        max_length: int = 32768,
        length_step: int = None,
        iterations: int = 20,
        num_keys: int = 1,
        aggressive_memory: bool = True,
        save_dir: str = "results/passkey",
        save_file: str = None,
    ):
        """
        Initialize evaluator

        Args:
            model: Loaded model object
            tokenizer: Tokenizer object
            restrict_tokens (bool): Whether to restrict output tokens (digits only), default is True
            data_mode (str): Data mode, "synthetic" or "real"
            dataset_name (str): Dataset name, only used for real mode
            split (str): Dataset split, only used for real mode
            min_length (int): Minimum token count, default is 2048
            max_length (int): Maximum token count, default is 8192
            length_step (int): Length step size; if None, uses exponential growth (multiply by 2 each time)
            iterations (int): Number of test iterations, default is 20
            num_keys (int): Number of pass keys, default is 1
            aggressive_memory (bool): Whether to clear memory after each iteration, default is True
            save_dir (str): Save directory, default is "results/passkey"
            save_file (str): Save filename, default is None
        """
        self.model = model
        self.tokenizer = tokenizer
        self.restrict_tokens = restrict_tokens
        self.data_mode = data_mode
        self.dataset_name = dataset_name
        self.split = split
        self.min_length = min_length
        self.max_length = max_length
        self.length_step = length_step
        self.iterations = iterations
        self.num_keys = num_keys
        self.aggressive_memory = aggressive_memory

        self.length_list = self._generate_length_list()
        self.data_loader = self._create_data_loader(
            data_mode,
            dataset_name,
            split,
            tokenizer,
            self.length_list[0],
        )
        self.pipe = self._setup_pipeline()

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_file = (
            save_file if save_file else f"{time.strftime('%Y%m%d-%H%M%S')}.json"
        )

    def _generate_length_list(self):
        """
        Generate a list of lengths for evaluating the model at different context lengths.

        If length_step is not None, grows at fixed step size (e.g., 2048, 2560, 3072, 3584...)
        If length_step is None, uses exponential growth (e.g., 2048, 4096, 8192...)

        Returns:
            list: List of lengths, e.g., [2048, 4096, 8192]
        """
        if self.length_step is not None:
            tokens = [
                x for x in range(self.min_length, self.max_length + 1, self.length_step)
            ]
        else:
            tokens = [self.min_length]
            while tokens[-1] < self.max_length:
                point = tokens[-1] * 2
                if point <= self.max_length:
                    tokens.append(point)
                else:
                    break
        return tokens

    def _create_data_loader(
        self,
        data_mode: str,
        dataset_name: str,
        split: str,
        tokenizer,
        length: int,
    ):
        """
        Create data loader

        Args:
            data_mode (str): Data mode
            dataset_name (str): Dataset name
            split (str): Dataset split
            tokenizer: Tokenizer object
            length (int): Target token count

        Returns:
            BaseDataLoader: Data loader object
        """
        if data_mode == "synthetic":
            return SyntheticDataLoader(tokenizer, length)
        else:
            return RealDataLoader(dataset_name, split, tokenizer, length)

    def _setup_pipeline(self):
        """
        Set up text generation pipeline

        If restrict_tokens is True, restrict model to only output digit tokens
        """
        if self.restrict_tokens:
            vocab = self.tokenizer.get_vocab()

            escape_chars = "▁Ġ"

            digit_tokens = [
                vocab[a] for a in vocab.keys() if a.lstrip(escape_chars).isdigit()
            ]
            digit_tokens.append(self.tokenizer.eos_token_id)
            extra = [
                vocab[a] for a in vocab.keys() if a.strip(" \n" + escape_chars) == ""
            ]
            digit_tokens.extend(extra)

            mask = torch.ones(self.tokenizer.vocab_size, dtype=torch.bool)
            mask[digit_tokens] = 0

            def filter_digits(module, input, output):
                output.logits[..., mask[: output.logits.size(-1)]] = -1e4

            self.model.register_forward_hook(filter_digits)
            print(f"Decoding restricted to {len(digit_tokens)} tokens.")

        self.tokenizer.model_max_length = self.max_length
        self.model.config.max_position_embeddings = self.max_length
        return pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def evaluate_sample(self):
        """
        Evaluate a single sample's pass key

        Returns:
            dict: Dictionary containing evaluation results:
                - prompt_text (str): Input prompt text
                - num_tokens (int): Token count of input text
                - pass_keys (list): Generated pass key list
                - target (int/str): Target pass key (number) or raw response
                - correct (bool): Whether target pass key was successfully extracted
        """
        prompt_text, pass_keys, target = self.data_loader.generate_prompt(self.num_keys)

        response = self.pipe(
            prompt_text,
            num_return_sequences=1,
            max_new_tokens=10,
            do_sample=False,
        )[0]["generated_text"][len(prompt_text) :]

        try:
            pass_key = int(re.search(r"\d+", response).group())
        except:
            pass_key = response[:20]

        if isinstance(self.data_loader, RealDataLoader):
            correct = str(pass_key).startswith(str(target))
        else:
            correct = pass_key == target

        result = {
            "prompt_text": prompt_text,
            "num_tokens": len(self.tokenizer.encode(prompt_text)),
            "pass_keys": pass_keys,
            "target": target,
            "correct": correct,
        }

        return result

    def evaluate(self):
        """
        Run needle-in-a-haystack evaluation across all configured lengths.

        Returns:
            dict: Evaluation result dictionary containing:
                - lengths (list): List of evaluated lengths
                - success_rates (list): List of success rates at corresponding lengths
        """
        success_rates = []

        pbar = tqdm(
            total=len(self.length_list),
            desc="Passkey Evaluation",
            leave=False,
        )

        for length in self.length_list:
            self.data_loader.set_length(length)

            results = []
            success_count = 0

            for _ in trange(self.iterations, desc=f"Length {length}", leave=False):
                result = self.evaluate_sample()
                success_count += result["correct"]
                results.append(result)

            success_rate = success_count / self.iterations if self.iterations > 0 else 0
            success_rates.append(success_rate)

            if self.aggressive_memory:
                gc.collect()
                torch.cuda.empty_cache()

            pbar.set_postfix(length=length, success_rate=f"{success_rate:.4f}")
            pbar.update(1)

        pbar.close()

        save_path = os.path.join(self.save_dir, self.save_file)
        with open(save_path, "w") as f:
            json.dump(
                {
                    "lengths": self.length_list,
                    "success_rates": success_rates,
                },
                f,
                indent=4,
            )

        return {
            "lengths": self.length_list,
            "success_rates": success_rates,
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


def add_args_passkey(parser):
    parser.add_argument("--num-keys", type=int, default=5, help="Number of pass keys.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of test iterations.",
    )
    parser.add_argument(
        "--data-mode",
        type=str,
        default="real",
        choices=["synthetic", "real"],
        help="Data mode.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="konwoo/RedPajama-Data-1T-Sample-subset1000",
        help="Dataset name.",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split.")
    parser.add_argument("--length-step", type=int, default=None, help="Length step.")
    parser.add_argument(
        "--aggressive-memory",
        type=bool,
        default=True,
        help="Whether to use aggressive memory.",
    )
    parser.add_argument(
        "--restrict-tokens",
        type=bool,
        default=True,
        help="Whether to restrict output tokens to digits only.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/passkey",
        help="Save directory.",
    )
    parser.add_argument(
        "--save-file",
        type=str,
        default=None,
        help="Save filename.",
    )
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args_model(parser)
    parser = add_args_passkey(parser)
    args = parser.parse_args()

    model, config = load_model(args)
    tokenizer = load_tokenizer(args)

    args.save_file = args.save_file or generate_save_filename(args)

    evaluator = PasskeyEvaluator(
        model=model,
        tokenizer=tokenizer,
        restrict_tokens=args.restrict_tokens,
        num_keys=args.num_keys,
        iterations=args.iterations,
        data_mode=args.data_mode,
        dataset_name=args.dataset_name,
        split=args.split,
        min_length=args.min_length,
        max_length=args.max_length,
        length_step=args.length_step,
        aggressive_memory=args.aggressive_memory,
        save_dir=args.save_dir,
        save_file=args.save_file,
    )
    results = evaluator.evaluate()
    print(results)
