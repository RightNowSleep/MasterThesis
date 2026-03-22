import gc
import os
import re
import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import pipeline
import argparse
import json
import time
from typing import Optional

from models.model_loader import load_model, load_tokenizer, add_args_model


# ---------------------------------------------------------------------------
# Prompt templates — text (legacy) mode
# ---------------------------------------------------------------------------

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

PROMPT_0SHOT_NO_CONTEXT = """What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

What is the single, most likely answer choice? Format your response as follows: "The correct answer is (insert answer here)"."""

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

# QuALITY dataset prompt for text mode
PROMPT_QUALITY_TEXT = """You are provided a story and a multiple-choice question with 4 possible answers (marked by A, B, C, D). Choose the best answer by writing its corresponding letter (either A, B, C, or D).

Story:
$DOC$

Question and Possible Answers:
$Q$
 (A) $C_A$
 (B) $C_B$
 (C) $C_C$
 (D) $C_D$

Format your response as follows: "The correct answer is (insert answer here)"."""

PROMPT_MAP = {
    "0shot": PROMPT_0SHOT,
    "0shot_no_context": PROMPT_0SHOT_NO_CONTEXT,
    "0shot_rag": PROMPT_0SHOT_RAG,
    "0shot_cot": PROMPT_0SHOT_COT,
    "0shot_cot_ans": PROMPT_0SHOT_COT_ANS,
    "quality": PROMPT_QUALITY_TEXT,
}

# ---------------------------------------------------------------------------
# Prompt templates — logit mode (prompt ends with "(" so model predicts A/B/C/D)
# ---------------------------------------------------------------------------

PROMPT_LOGIT_LONGBENCH_V2 = """Please read the following text and answer the question below.

<text>
{context}
</text>

What is the correct answer to this question: {question}
Choices:
(A) {choice_A}
(B) {choice_B}
(C) {choice_C}
(D) {choice_D}

The correct answer is ("""

PROMPT_LOGIT_QUALITY = """You are provided a story and a multiple-choice question with 4 possible answers (marked by A, B, C, D). Choose the best answer by writing its corresponding letter (either A, B, C, or D).

Story:
{story}

Question and Possible Answers:
{question}
 (A) {a}
 (B) {b}
 (C) {c}
 (D) {d}

Answer: ("""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHOICES = ["A", "B", "C", "D"]
ANSWER_PREFIX = "The correct answer is ("

# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class QualityEvaluator:
    r"""
    MCQ quality evaluator for long-context models.

    Supports two datasets
    ---------------------
    * **LongBench-v2** (``zai-org/LongBench-v2``) — default dataset, single split "train".
    * **QuALITY** (``emozilla/quality``) — SCROLLS/QuALITY benchmark, splits "train"/"validation".

    Scoring modes
    -------------
    * ``"logit"`` *(default)* — reads the raw next-token logit for each of the four
      choice letters.  Robust against instruction-following failures; never produces
      a parse error.  Not compatible with COT/RAG (those require text generation).
    * ``"text"`` — generates up to 128 tokens and extracts the answer via regex.
      Required for COT / RAG modes; prone to high parse-failure rates on base models.

    Attributes
    ----------
    model, tokenizer, dataset_name, subset, split, limit, max_length,
    aggressive_memory, cot, no_context, rag, scoring_mode, save_dir, save_file
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
        scoring_mode: str = "logit",
        save_dir: str = "results/quality",
        save_file: str = None,
    ):
        r"""
        Initialize the quality evaluator.

        Args:
            model: The model to evaluate.
            tokenizer: The tokenizer to use for the model.
            dataset_name: The name of the dataset to use.
            subset: The subset of the dataset to use.
            split: The split of the dataset to use.
            limit: The number of examples to evaluate.
            max_length: The maximum length of the input sequence.
            aggressive_memory: Whether to use aggressive memory.
            cot: Whether to use COT mode.
            no_context: Whether to evaluate without the long context.
            rag: The number of retrieved chunks to use.
            scoring_mode: The scoring mode to use.
            save_dir: The directory to save the results.
            save_file: The filename to save the results to.
        """
        # ── Validate scoring_mode / mode combinations ─────────────────── #
        assert scoring_mode in (
            "logit",
            "text",
        ), "scoring_mode must be 'logit' or 'text'"
        if scoring_mode == "logit" and (cot or rag > 0):
            print(
                "[WARNING] COT and RAG modes require text generation; "
                "overriding scoring_mode to 'text'."
            )
            scoring_mode = "text"

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
        self.scoring_mode = scoring_mode

        # Infer dataset type from name
        self.dataset_type = self._infer_dataset_type(dataset_name)

        # Determine prompt type for text mode
        if scoring_mode == "text":
            if self.rag > 0:
                self.prompt_type = "0shot_rag"
            elif self.cot:
                self.prompt_type = "0shot_cot"
            elif self.no_context:
                self.prompt_type = "0shot_no_context"
            elif self.dataset_type == "quality":
                self.prompt_type = "quality"
            else:
                self.prompt_type = "0shot"
        else:
            self.prompt_type = None

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Choice token IDs (A / B / C / D)
        self.choice_tokens = self._get_choice_tokens()

        # Load dataset
        self.dataset = self._load_and_preprocess_dataset()

        # Set up generation pipeline only for text mode
        self.pipe = self._setup_pipeline() if scoring_mode == "text" else None

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_file = (
            save_file if save_file else f"{time.strftime('%Y%m%d-%H%M%S')}.json"
        )

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _infer_dataset_type(self, dataset_name: str) -> str:
        """
        Infer the dataset type from the name.

        Args:
            dataset_name: The name of the dataset to use.

        Returns:
            dataset_type: The dataset type, either 'quality' or 'longbench_v2'.
        """
        if "quality" in dataset_name.lower():
            return "quality"
        return "longbench_v2"

    def _get_choice_tokens(self) -> list:
        """
        Get the choice letters ('A choice letters ('A'/'B'/'C'/'D').

        Returns:
            choice_tokens: The token IDs for the choice letters ('A'/'B'/'C'/'D').
        """
        return [self.tokenizer.encode(c, add_special_tokens=False)[0] for c in CHOICES]

    def _setup_pipeline(self) -> pipeline:
        """
        Set up the text-generation pipeline.

        Returns:
            pipe: The text-generation pipeline.
        """
        return pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def _load_and_preprocess_dataset(self):
        """
        Load and optionally limit the dataset.

        Returns:
            dataset: The loaded dataset.
        """
        print(
            f"Loading dataset {self.dataset_name} "
            f"(subset: {self.subset}, split: {self.split})..."
        )
        dataset = load_dataset(self.dataset_name, self.subset, split=self.split)
        if self.limit:
            dataset = dataset.select(range(min(self.limit, len(dataset))))
        return dataset

    def _get_answer_label(self, sample) -> str:
        """
        Get the correct answer letter ('A'/'B'/'C'/'D') for a sample.

        Args:
            sample: The sample from the dataset.

        Returns:
            answer_label: The correct answer letter ('A'/'B'/'C'/'D').
        LongBench-v2 stores the answer as a letter string already.
        QuALITY stores the answer as an integer index 0-3.
        """
        if self.dataset_type == "quality":
            return CHOICES[int(sample["answer"])]
        return sample["answer"]  # already 'A'/'B'/'C'/'D'

    # ------------------------------------------------------------------ #
    # Logit scoring
    # ------------------------------------------------------------------ #

    def _prepare_logit_prompt(self, sample) -> str:
        """Build the prompt for logit scoring (ends with '(' for the model to complete)."""
        if self.dataset_type == "quality":
            options = sample.get("options", ["", "", "", ""])
            context = sample.get("article", "")
            raw = PROMPT_LOGIT_QUALITY.format(
                story=context.strip(),
                question=sample["question"].strip(),
                a=options[0].strip(),
                b=options[1].strip(),
                c=options[2].strip(),
                d=options[3].strip(),
            )
        else:  # longbench_v2
            raw = PROMPT_LOGIT_LONGBENCH_V2.format(
                context=sample.get("context", "").strip(),
                question=sample["question"].strip(),
                choice_A=sample["choice_A"].strip(),
                choice_B=sample["choice_B"].strip(),
                choice_C=sample["choice_C"].strip(),
                choice_D=sample["choice_D"].strip(),
            )

        if self.max_length is not None:
            input_ids = self.tokenizer.encode(raw)
            if len(input_ids) > self.max_length:
                half = self.max_length // 2
                input_ids = input_ids[:half] + input_ids[-half:]
                raw = self.tokenizer.decode(input_ids, skip_special_tokens=True)

        return raw

    @torch.no_grad()
    def _evaluate_logit(self, sample) -> dict:
        """
        Score a sample via first-token logits.

        Args:
            sample: The sample from the dataset.

        Returns:
            dict: A dictionary with the model's prediction, correctness, and response.
                - prediction: The model's prediction ('A'/'B'/'C'/'D').
                - correct: Whether the model's prediction is correct.
                - parse_failed: Whether the model's response failed to be parsed.
                - response: The model's response, formatted as "[logit] A={v:.2f} B={v:.2f} C={v:.2f} D={v:.2f}]"
                    - False otherwise.
        """
        prompt = self._prepare_logit_prompt(sample)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        first_device = next(self.model.parameters()).device
        input_ids = inputs.input_ids.to(first_device)
        attention_mask = inputs.attention_mask.to(first_device)

        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # output.scores is a tuple of length 1 (one generated token);
        # scores[0] has shape [batch=1, vocab_size]
        scores = output.scores[0][0]
        choice_logits = [scores[t].float().cpu().item() for t in self.choice_tokens]
        prediction_idx = int(np.argmax(choice_logits))
        prediction = CHOICES[prediction_idx]

        answer = self._get_answer_label(sample)
        correct = prediction == answer

        logit_str = " ".join(f"{c}={v:.2f}" for c, v in zip(CHOICES, choice_logits))
        return {
            "prediction": prediction,
            "correct": correct,
            "parse_failed": False,
            "response": f"[logit] {logit_str}",
        }

    # ------------------------------------------------------------------ #
    # Text scoring (legacy)
    # ------------------------------------------------------------------ #

    def _prepare_text_prompt(self, sample, cot_response: str = None) -> str:
        """
        Build the prompt for text-generation scoring.

        Args:
            sample: The sample from the dataset.
            cot_response: The COT response to use for the prompt, if None otherwise.

        Returns:
            str: The prompt prompt string.
        """
        prompt_type = "0shot_cot_ans" if cot_response else self.prompt_type

        # For QuALITY dataset, normalise field names to the shared template variables
        if self.dataset_type == "quality":
            options = sample.get("options", ["", "", "", ""])
            context = sample.get("article", "")
            choice_a, choice_b, choice_c, choice_d = (
                options[0],
                options[1],
                options[2],
                options[3],
            )
            prompt_type_actual = "quality" if prompt_type == "0shot" else prompt_type
        else:
            context = sample.get("context", "")
            choice_a = sample.get("choice_A", "")
            choice_b = sample.get("choice_B", "")
            choice_c = sample.get("choice_C", "")
            choice_d = sample.get("choice_D", "")
            prompt_type_actual = prompt_type

        if prompt_type_actual == "0shot_rag" and self.rag > 0:
            retrieved = sample.get("retrieved_context", [])[: self.rag]
            retrieved = sorted(retrieved, key=lambda x: x["c_idx"])
            context = "\n\n".join(
                f"Retrieved chunk {idx + 1}: {x['content']}"
                for idx, x in enumerate(retrieved)
            )

        prompt_template = PROMPT_MAP.get(prompt_type_actual, PROMPT_0SHOT)
        prompt = (
            prompt_template.replace("$DOC$", context.strip())
            .replace("$Q$", sample["question"].strip())
            .replace("$C_A$", choice_a.strip())
            .replace("$C_B$", choice_b.strip())
            .replace("$C_C$", choice_c.strip())
            .replace("$C_D$", choice_d.strip())
        )

        if cot_response:
            prompt = prompt.replace("$COT$", cot_response.strip())

        if self.max_length is not None:
            input_ids = self.tokenizer.encode(prompt)
            if len(input_ids) > self.max_length:
                half = self.max_length // 2
                input_ids = input_ids[:half] + input_ids[-half:]
                prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)

        return prompt

    def _extract_answer(self, response: str) -> Optional[str]:
        """
        Extract the answer from the model's response.

        Args:
            response: The model's response, formatted as "The correct answer is (A-D)" or "The correct answer is A-D".

        Returns:
            Optional[str]: The extracted answer ('A'/'B'/'C'/'D').
        """
        response = response.replace("*", "")
        m = re.search(r"The correct answer is \(([A-D])\)", response)
        if m:
            return m.group(1)
        m = re.search(r"The correct answer is ([A-D])", response)
        if m:
            return m.group(1)
        return None

    @torch.no_grad()
    def _evaluate_text(self, sample) -> dict:
        """
        Score a sample via text generation + regex parsing.

        Args:
            sample: The sample from the dataset.

        Returns:
            dict: A dictionary with the model's prediction, correctness, and response.
                - prediction: The model's prediction ('A'/'B'/'C'/'D').
                - correct: Whether the model's prediction is correct.
                - parse_failed: Whether the model's response failed to be parsed.
                - response: The model's response, formatted as "The correct answer is (A-D)".
                - response_cot: The model's response, formatted as "The correct answer is (A-D)".
                    - None otherwise.
        """
        prompt = self._prepare_text_prompt(sample)
        cot_response = None

        if self.cot:
            cot_output = self.pipe(
                prompt,
                num_return_sequences=1,
                max_new_tokens=1024,
                do_sample=False,
            )[0]["generated_text"][len(prompt) :]
            cot_response = cot_output.strip()
            prompt = self._prepare_text_prompt(sample, cot_response=cot_response)

        output = self.pipe(
            prompt,
            num_return_sequences=1,
            max_new_tokens=128,
            do_sample=False,
        )[0]["generated_text"][len(prompt) :]

        response = output.strip()
        prediction = self._extract_answer(response)
        answer = self._get_answer_label(sample)
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

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def evaluate_sample(self, sample) -> dict:
        """
        Dispatch to logit or text scoring based on self.scoring_mode.

        Args:
            sample: The sample from the dataset.

        Returns:
            dict: A dictionary with the model's prediction, correctness, and response.
                - prediction: The model's prediction ('A'/'B'/'C'/'D').
                - correct: Whether the model's prediction is correct.
                - parse_failed: Whether the model's response failed to be parsed.
                - response: The model's response, formatted as "The correct answer is (A-D)".
                - response_cot: The model's response, formatted as "The correct answer is (A-D)".
                    - None otherwise.
        """
        if self.scoring_mode == "logit":
            return self._evaluate_logit(sample)
        else:
            return self._evaluate_text(sample)

    def evaluate(self) -> dict:
        r"""
        Run the full evaluation over the loaded dataset.

        Returns
        -------
        dict
            summary dict with keys: accuracy, correct_count, total_samples,
            wrong_count, parse_fail_count, skip_count.
        """
        correct_count = 0
        total_samples = 0
        wrong_count = 0
        parse_fail_count = 0
        skip_count = 0
        sample_records = []

        desc = f"Quality Evaluation [{self.scoring_mode}] ({self.dataset_type})"
        pbar = tqdm(total=len(self.dataset), desc=desc)

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

            # Build per-sample record
            if self.dataset_type == "quality":
                options = sample.get("options", ["", "", "", ""])
                record = {
                    "question": sample.get("question", ""),
                    "answer": self._get_answer_label(sample),
                    "pred": eval_result["prediction"],
                    "judge": eval_result["correct"],
                    "parse_failed": eval_result["parse_failed"],
                    "response": eval_result["response"],
                    "context_preview": sample.get("article", "")[:1000],
                }
            else:
                record = {
                    "question": sample.get("question", ""),
                    "answer": self._get_answer_label(sample),
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

        accuracy = correct_count / total_samples * 100 if total_samples > 0 else 0.0
        summary = {
            "accuracy": accuracy,
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


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def generate_save_filename(model_name, config) -> str:
    """
    Generate the filename filename for saving the evaluation results.

    Args:
        model_name: The name of the model.
        config: The configuration object.

    Returns:
        str: The filename.
    """
    model_name = model_name.split("/")[-1]
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling["type"] if rope_scaling else "none"
    parts = [model_name, rope_type]
    if rope_type != "none":
        factor = getattr(config, "factor", None)
        dynamic = getattr(config, "dynamic", False)
        if factor is not None:
            parts.append(f"factor{str(factor).replace('.', '_')}")
        elif dynamic:
            parts.append("dynamic")
    return "_".join(parts) + ".json"


def add_args_quality(parser):
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="zai-org/LongBench-v2",
        choices=["zai-org/LongBench-v2", "emozilla/quality"],
        help=(
            "Dataset name. Supported: "
            "'zai-org/LongBench-v2' (default), 'emozilla/quality' (QuALITY/SCROLLS)."
        ),
    )
    parser.add_argument("--subset", type=str, default=None, help="Dataset subset")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split. LongBench-v2: 'train'. QuALITY: 'train' or 'validation'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--cot",
        action="store_true",
        help="Use Chain-of-Thought (text mode only)",
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Evaluate without context (text mode only)",
    )
    parser.add_argument(
        "--rag",
        type=int,
        default=0,
        help="Number of RAG chunks to use (text mode only, LongBench-v2 only)",
    )
    parser.add_argument(
        "--aggressive-memory",
        action="store_true",
        help="Clear GPU cache after each sample",
    )
    parser.add_argument(
        "--scoring-mode",
        type=str,
        default="logit",
        choices=["logit", "text"],
        help=(
            "Scoring mode. "
            "'logit' (default): reads first-token logits for A/B/C/D — robust, no parse failures. "
            "'text' (legacy): generates text and parses with regex — required for COT/RAG."
        ),
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
    print(f"\n=== Evaluation Complete ===")
    print(f"Dataset       : {args.dataset_name}")
    print(f"Scoring mode  : {args.scoring_mode}")
    print(f"Accuracy      : {results['accuracy']:.2f}%")
    print(f"Correct       : {results['correct_count']} / {results['total_samples']}")
    print(f"Wrong         : {results['wrong_count']}")
    print(f"Parse Failed  : {results['parse_fail_count']}")
    print(f"Skipped       : {results['skip_count']}")
