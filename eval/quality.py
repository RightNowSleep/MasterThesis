import gc
import os
import re
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import pipeline
import argparse
import json
import time
from models.model_loader import load_model, load_tokenizer, add_args_model


# (1) Basic template
PROMPT_0SHOT = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: "The correct answer is (insert answer here)"."""

# (2) No context template
PROMPT_0SHOT_NO_CONTEXT = """What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

What is the single, most likely answer choice? Format your response as follows: "The correct answer is (insert answer here)"."""

# (3) RAG template
PROMPT_0SHOT_RAG = """Please read the following retrieved text chunks and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: "The correct answer is (insert answer here)"."""

# (4) COT reasoning template
PROMPT_0SHOT_COT = """Please read the following text and answer the questions below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Let's think step by step:"""

# (5) COT answer template
PROMPT_0SHOT_COT_ANS = """Please read the following text and answer the questions below.

The text is too long and omitted here.

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Let's think step by step: $COT$

Based on the above, what is the single, most likely answer choice? Format your response as follows: "The correct answer is (insert answer here)"."""

PROMPT_MAP = {
    "0shot": PROMPT_0SHOT,
    "0shot_no_context": PROMPT_0SHOT_NO_CONTEXT,
    "0shot_rag": PROMPT_0SHOT_RAG,
    "0shot_cot": PROMPT_0SHOT_COT,
    "0shot_cot_ans": PROMPT_0SHOT_COT_ANS,
}

CHOICES = ["A", "B", "C", "D"]
ANSWER_PREFIX = "The correct answer is ("


class QualityEvaluator:
    r"""
    QualityEvaluator for computing QA accuracy of long-context models on the LongBench v2 dataset.

    Attributes:
        model: Language model for computing logits
        tokenizer: Tokenizer for processing text
        dataset_name: Dataset name
        subset: Dataset subset name
        split: Dataset split
        limit: Limit on number of samples to evaluate
        max_length: Maximum context token length
        aggressive_memory: Whether to enable aggressive memory management
        cot: Whether to use Chain of Thought
        no_context: Whether to not use context
        rag: Number of retrieval chunks to use in RAG mode
        save_dir: Directory to save results
        save_file: Filename to save results
    """

    def __init__(
        self,
        model,
        tokenizer,
        dataset_name: str = "zai-org/LongBench-v2",
        subset: str = None,
        split: str = "train",
        limit: int = None,
        max_length: int = None,
        aggressive_memory: bool = True,
        cot: bool = False,
        no_context: bool = False,
        rag: int = 0,
        save_dir: str = "results/quality",
        save_file: str = None,
    ):
        r"""
        Initialize the QualityEvaluator.

        Args:
            model: Language model (required), pretrained model for evaluation
            tokenizer: Tokenizer (required), tokenizer for processing text
            dataset_name (str, optional): Dataset name, default is "zai-org/LongBench-v2"
            subset (str, optional): Dataset subset/config name, e.g., "narrativeqa", "hotpotqa", etc.
            split (str, optional): Dataset split, default is "train"
            limit (int, optional): Limit on number of samples to evaluate
            max_length (int, optional): Maximum context token length
            aggressive_memory (bool, optional): Whether to enable aggressive memory management, default is True
            cot (bool, optional): Whether to use Chain of Thought, default is False
            no_context (bool, optional): Whether to not use context, default is False
            rag (int, optional): Number of retrieval chunks to use in RAG mode, default is 0
            save_dir (str, optional): Directory to save results, default is "results/quality"
            save_file (str, optional): Filename to save results, default is None
        """
        self.model = model.eval()
        self.config = model.config
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.limit = limit
        self.max_length = max_length
        self.aggressive_memory = aggressive_memory
        self.cot = cot
        self.no_context = no_context
        self.rag = rag

        # Determine prompt type based on running mode
        if self.rag > 0:
            self.prompt_type = "0shot_rag"
        elif self.cot:
            self.prompt_type = "0shot_cot"
        elif self.no_context:
            self.prompt_type = "0shot_no_context"
        else:
            self.prompt_type = "0shot"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.dataset = self._load_and_preprocess_dataset()
        self.pipe = self._setup_pipeline()

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_file = (
            save_file if save_file else f"{time.strftime('%Y%m%d-%H%M%S')}.json"
        )

    def _setup_pipeline(self):
        r"""
        Set up text generation pipeline

        Returns:
            pipeline: Text generation pipeline object
        """
        return pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def _load_and_preprocess_dataset(self):
        """Load dataset"""
        print(
            f"Loading dataset {self.dataset_name} (Subset: {self.subset}, Split: {self.split})..."
        )
        dataset = load_dataset(self.dataset_name, self.subset, split=self.split)

        if self.limit:
            dataset = dataset.select(range(min(self.limit, len(dataset))))

        return dataset

    def _prepare_prompt(self, sample, cot_response: str = None):
        r"""
        Prepare the model prompt with truncation strategy.

        Strategy:
        1. Build prompt from template
        2. Calculate prompt token length
        3. Truncate prompt by removing half from front and half from back
        4. Concatenate final prompt

        Args:
            sample: Data sample
            cot_response (str, optional): COT response, used for COT_ANS mode

        Returns:
            str: Prepared prompt text
        """
        prompt_type = "0shot_cot_ans" if cot_response else self.prompt_type
        prompt_template = PROMPT_MAP.get(prompt_type, PROMPT_0SHOT)

        context = sample.get("context", "")

        if prompt_type == "0shot_rag" and self.rag > 0:
            retrieved = sample["retrieved_context"][: self.rag]
            retrieved = sorted(retrieved, key=lambda x: x["c_idx"])
            context = "\n\n".join(
                [
                    f"Retrieved chunk {idx + 1}: {x['content']}"
                    for idx, x in enumerate(retrieved)
                ]
            )

        prompt = (
            prompt_template.replace("$DOC$", context.strip())
            .replace("$Q$", sample["question"].strip())
            .replace("$C_A$", sample["choice_A"].strip())
            .replace("$C_B$", sample["choice_B"].strip())
            .replace("$C_C$", sample["choice_C"].strip())
            .replace("$C_D$", sample["choice_D"].strip())
        )

        if cot_response:
            prompt = prompt.replace("$COT$", cot_response.strip())

        if self.max_length is not None:
            input_ids = self.tokenizer.encode(prompt)
            if len(input_ids) > self.max_length:
                input_ids = (
                    input_ids[: self.max_length // 2]
                    + input_ids[-self.max_length // 2 :]
                )
                prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)

        return prompt

    def _extract_answer(self, response: str):
        r"""
        Extract answer from model response

        Args:
            response (str): Full response text from the model

        Returns:
            Optional[str]: Extracted answer (A/B/C/D), or None if extraction fails
        """
        response = response.replace("*", "")
        match = re.search(r"The correct answer is \(([A-D])\)", response)
        if match:
            return match.group(1)
        match = re.search(r"The correct answer is ([A-D])", response)
        if match:
            return match.group(1)
        return None

    @torch.no_grad()
    def evaluate_sample(self, sample):
        r"""
        Evaluate a single sample

        Args:
            sample: Data sample

        Returns:
            dict: Evaluation result containing prediction, correct, response, etc.;
                  returns None if prompt preparation fails
        """
        prompt = self._prepare_prompt(sample)
        if prompt is None:
            return None

        cot_response = None

        if self.cot:
            cot_output = self.pipe(
                prompt,
                num_return_sequences=1,
                max_new_tokens=1024,
                do_sample=False,
            )[0]["generated_text"][len(prompt) :]

            cot_response = cot_output.strip()

            prompt = self._prepare_prompt(sample, cot_response=cot_response)
            if prompt is None:
                return None

        output = self.pipe(
            prompt,
            num_return_sequences=1,
            max_new_tokens=128,
            do_sample=False,
        )[0]["generated_text"][len(prompt) :]

        response = output.strip()
        prediction = self._extract_answer(response)

        answer = sample["answer"]
        correct = prediction is not None and answer == prediction

        result = {
            "prediction": prediction,
            "correct": correct,
            "parse_failed": prediction is None,
            "response": response,
        }

        if cot_response is not None:
            result["response_cot"] = cot_response

        return result

    def evaluate(self):
        r"""
        Execute evaluation

        Returns:
            dict: Evaluation results containing accuracy, correct count, total count, etc.
                - accuracy (float): Accuracy, range [0, 100]
                - correct_count (int): Number of correctly predicted samples
                - total_samples (int): Total number of samples
                - wrong_count (int): Number of samples with wrong answers
                - parse_fail_count (int): Number of samples where answer could not be parsed
                - skip_count (int): Number of samples skipped due to prompt preparation failure
        """
        correct_count = 0
        total_samples = 0
        wrong_count = 0
        parse_fail_count = 0
        skip_count = 0  # Skip count due to prompt preparation failure
        sample_records = []

        pbar = tqdm(
            total=len(self.dataset),
            desc=f"Quality Evaluation ({self.prompt_type})",
        )

        for sample in self.dataset:
            eval_result = self.evaluate_sample(sample)

            if eval_result is None:
                skip_count += 1
                pbar.update(1)
                continue

            total_samples += 1
            if eval_result["correct"]:
                correct_count += 1
            elif eval_result["parse_failed"]:
                parse_fail_count += 1
            else:
                wrong_count += 1

            record = {
                "question": sample.get("question", ""),
                "answer": sample.get("answer", ""),
                "pred": eval_result["prediction"],
                "judge": eval_result["correct"],
                "parse_failed": eval_result["parse_failed"],
                "response": eval_result["response"],
                "context_preview": sample.get("context", "")[:1000],
            }
            if "response_cot" in eval_result:
                record["response_cot"] = eval_result["response_cot"]
            sample_records.append(record)

            acc = correct_count / total_samples * 100 if total_samples > 0 else 0
            pbar.set_postfix(acc=f"{acc:.1f}%")
            pbar.update(1)

            if self.aggressive_memory:
                gc.collect()
                torch.cuda.empty_cache()

        pbar.close()

        summary = {
            "accuracy": (
                correct_count / total_samples * 100 if total_samples > 0 else 0.0
            ),
            "correct_count": correct_count,
            "total_samples": total_samples,
            "wrong_count": wrong_count,
            "parse_fail_count": parse_fail_count,
            "skip_count": skip_count,
        }

        save_path = os.path.join(self.save_dir, self.save_file)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(
                {"summary": summary, "samples": sample_records},
                f,
                ensure_ascii=False,
                indent=4,
            )

        return summary


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


def add_args_quality(parser):
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="zai-org/LongBench-v2",
        help="Dataset name",
    )
    parser.add_argument("--subset", type=str, default=None, help="Dataset subset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--limit", type=int, default=10, help="Dataset limit")
    parser.add_argument("--cot", action="store_true", help="Whether to use COT")
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Whether to not use context",
    )
    parser.add_argument(
        "--rag",
        type=int,
        default=0,
        help="Whether to use RAG, 0 means not using",
    )
    parser.add_argument(
        "--aggressive-memory",
        action="store_true",
        help="Aggressive memory management",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/quality",
        help="Save directory",
    )
    parser.add_argument("--save-file", type=str, default=None, help="Save filename")
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quality evaluation")
    parser = add_args_quality(parser)
    parser = add_args_model(parser)
    args = parser.parse_args()

    model, config = load_model(args)
    tokenizer = load_tokenizer(args)

    args.save_file = args.save_file or generate_save_filename(args)

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
        save_dir=args.save_dir,
        save_file=args.save_file,
    )

    results = evaluator.evaluate()
    print(f"\n=== Evaluation Complete ===")
    print(f"Accuracy      : {results['accuracy']:.2f}%")
    print(f"Correct       : {results['correct_count']} / {results['total_samples']}")
    print(f"Wrong         : {results['wrong_count']}")
    print(f"Parse Failed  : {results['parse_fail_count']}")
    print(f"Skipped       : {results['skip_count']}")
