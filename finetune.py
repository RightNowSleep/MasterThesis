"""Supervised fine-tuning (SFT) script with QLoRA and DeepSpeed support.

Implements an end-to-end SFT pipeline for instruction-tuning large language models
using Hugging Face TRL's :class:`~trl.SFTTrainer`. Supports:

    - **QLoRA**: 4-bit or 8-bit quantization via BitsAndBytes for memory-efficient
      fine-tuning on consumer GPUs.
    - **LoRA**: Optional Low-Rank Adaptation wrapping for parameter-efficient SFT.
    - **DeepSpeed**: Integration with DeepSpeed ZeRO offloading for multi-GPU scaling.
    - **Chat template formatting**: Automatic message-to-text conversion via the
      tokenizer's chat template.
    - **Multiple RoPE types**: Linear, NTK-aware, YaRN, and dynamic RoPE scaling
      inherited from the shared model loader.

Typical usage::

    python finetune.py \\
        --model-name meta-llama/Llama-2-7b-chat-hf \\
        --rope-type linear --rope-factor 4.0 \\
        --quantization 4bit --use-lora \\
        --dataset HuggingFaceH4/ultrachat_200k
"""

import argparse
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from transformers import BitsAndBytesConfig

from models.model_loader import load_model, load_tokenizer, add_args_model


def add_args_finetune(parser: argparse.ArgumentParser):
    """Add SFT-specific hyperparameters to the argument parser.

    Registers CLI flags for model output, training epochs, batch size, learning rate,
    LoRA configuration, quantization mode, dataset selection, and DeepSpeed settings.
    Model/tokenizer/RoPE arguments are handled separately by :func:`add_args_model`.

    Args:
        parser: Argument parser instance to augment with SFT arguments.

    Returns:
        argparse.ArgumentParser: The same parser with additional SFT arguments registered.

    Added Arguments:
        --output-dir: Directory for saving the fine-tuned model.
        --num-train-epochs: Number of full passes over the training dataset.
        --per-device-train-batch-size: Batch size per GPU/device.
        --gradient-accumulation-steps: Steps before optimizer update.
        --learning-rate: Peak learning rate for the optimizer.
        --max-seq-length: Maximum sequence length passed to SFTTrainer.
        --optim: Optimizer name (adamw_torch, adamw_bnb_8bit, paged_adamw_8bit).
        --deepspeed: Path to DeepSpeed JSON config file (None disables).
        --use-lora: Enable LoRA wrapping for parameter-efficient fine-tuning.
        --lora-r: LoRA rank (dimension of low-rank matrices).
        --lora-alpha: LoRA scaling factor (recommend >= lora_r).
        --lora-dropout: Dropout probability applied to LoRA layers.
        --quantization: Quantization mode — "4bit", "8bit", or "none".
        --dataset: HuggingFace dataset identifier for training data.
        --dataset-split: Dataset split to use (e.g., "train_sft").
    """
    parser.add_argument(
        "--output-dir",
        type=str,
        default="finetunes/finetune",
    )
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=8192,
        help="Maximum sequence length for SFTTrainer. Should be <= --max-length.",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="paged_adamw_8bit",
        choices=["adamw_torch", "adamw_bnb_8bit", "paged_adamw_8bit"],
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to DeepSpeed config JSON. Leave empty to disable.",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Wrap model with LoRA for parameter-efficient SFT.",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha. Recommend >= lora_r.",
    )
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--quantization",
        type=str,
        default="none",
        choices=["4bit", "8bit", "none"],
        help="Quantization mode. load_in_4bit/8bit flags are set automatically.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceH4/ultrachat_200k",
        help="HuggingFace dataset id for SFT.",
    )
    parser.add_argument("--dataset-split", type=str, default="train_sft")
    return parser


def main(args):
    """Execute the supervised fine-tuning pipeline.

    Pipeline steps:

        1. Derive experiment tag from ``--rope-type``.
        2. Load tokenizer and define chat-template formatting closure.
        3. Build BitsAndBytesConfig if quantization is requested.
        4. Patch args for compatibility with the unified model loader.
        5. Load model via :func:`models.model_loader.load_model`.
        6. Optionally wrap model with LoRA configuration.
        7. Load training dataset from Hugging Face Hub.
        8. Instantiate SFTTrainer with training arguments.
        9. Run training and save the final model.

    Args:
        args: Parsed command-line arguments containing model, SFT, LoRA,
            quantization, and dataset configurations.

    Returns:
        None
    """
    # ------------------------------------------------------------------
    # Derive experiment tag from --rope-type
    # ------------------------------------------------------------------
    _tag = args.rope_type

    # ------------------------------------------------------------------
    # Tokenizer: Load first so formatting_func can close over it
    # ------------------------------------------------------------------
    tokenizer = load_tokenizer(args)
    tokenizer.model_max_length = args.max_seq_length

    def formatting_func(example):
        """Convert a chat-messages example into a single formatted text string.

        Args:
            example: A dictionary containing a ``"messages"`` key with a list
                of role-content dictionaries.

        Returns:
            dict: Dictionary with a single ``"text"`` key holding the formatted string.
        """
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
        }

    # ------------------------------------------------------------------
    # Quantization: Build QLoRA-grade BitsAndBytesConfig
    # ------------------------------------------------------------------
    quantization_config = None
    if args.quantization in ("4bit", "8bit"):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=(args.quantization == "4bit"),
            load_in_8bit=(args.quantization == "8bit"),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print(f"Quantization: {args.quantization} (QLoRA / nf4)")

    # Align flags that load_model reads internally
    args.load_in_4bit = args.quantization == "4bit"
    args.load_in_8bit = args.quantization == "8bit"

    # Disable use_cache during training (conflicts with gradient checkpointing)
    args.use_cache = False
    # SFTTrainer creates its own LoRA; don't merge a pre-existing adapter
    args.adapter_path = None
    # SFTTrainer handles gradient checkpointing; avoid double-enabling
    args.gradient_checkpointing = False

    # ------------------------------------------------------------------
    # Model: Load via unified loader.
    # load_model returns (model, config); config is not needed here.
    # ------------------------------------------------------------------
    print("=== Loading model with unified model_loader ===")
    model, _ = load_model(args, quantization_config=quantization_config)

    # ------------------------------------------------------------------
    # LoRA: Configure parameter-efficient fine-tuning if enabled
    # ------------------------------------------------------------------
    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules="all-linear",
            bias="none",
            task_type="CAUSAL_LM",
        )
        print(
            f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}"
        )

    # ------------------------------------------------------------------
    # Dataset: Load training data from HuggingFace Hub
    # ------------------------------------------------------------------
    dataset = load_dataset(args.dataset, split=args.dataset_split)
    # dataset = dataset.select(range(20000))  # Uncomment for debug runs

    # ------------------------------------------------------------------
    # Trainer: Configure SFTTrainer with training arguments
    # ------------------------------------------------------------------
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        bf16=True,
        optim=args.optim,
        packing=True,
        dataset_num_proc=8,
        report_to="tensorboard",
        deepspeed=args.deepspeed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_func,
        args=sft_config,
    )

    print(
        f"=== Starting SFT | rope={args.rope_type} "
        f"| quant={args.quantization} | lora={args.use_lora} ==="
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Done. Model saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT with QLoRA + DeepSpeed")
    parser = add_args_model(parser)
    parser = add_args_finetune(parser)
    args = parser.parse_args()
    main(args)
