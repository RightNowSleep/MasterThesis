import argparse
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from transformers import BitsAndBytesConfig

from models.model_loader import load_model, load_tokenizer, add_args_model


def add_args_finetune(parser: argparse.ArgumentParser):
    r"""
    Add SFT-specific hyperparameters to the argument parser.

    Model, tokenizer, and RoPE arguments are inherited from
    add_args_model() in the model_loader module.

    Added Arguments:
        --output-dir: Directory to save the fine-tuned model.
        --num-train-epochs: Number of training epochs.
        --per-device-train-batch-size: Batch size per device.
        --gradient-accumulation-steps: Number of gradient accumulation steps.
        --learning-rate: Learning rate for the optimizer.
        --max-seq-length: Maximum sequence length for SFTTrainer.
        --optim: Optimizer type.
        --deepspeed: Path to DeepSpeed config JSON file.
        --use-lora: Whether to use LoRA for parameter-efficient fine-tuning.
        --lora-r: LoRA rank.
        --lora-alpha: LoRA alpha parameter.
        --lora-dropout: LoRA dropout rate.
        --quantization: Quantization mode (4bit, 8bit, none).
        --dataset: HuggingFace dataset ID for SFT.
        --dataset-split: Dataset split to use for training.
    """
    parser.add_argument("--output-dir", type=str, default="finetunes")
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
    parser.add_argument("--lora-alpha", type=int, default=32)
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
    r"""
    Main entry point for supervised fine-tuning (SFT) with QLoRA and DeepSpeed.

    Pipeline:
    1. Parse command-line arguments for model, SFT, and LoRA configurations.
    2. Load tokenizer and define chat-template formatting function.
    3. Build quantization config (4-bit or 8-bit) if requested.
    4. Load the model via the unified loader.
    5. Configure LoRA for parameter-efficient fine-tuning if enabled.
    6. Load and prepare the training dataset.
    7. Initialize SFTTrainer and start training.
    8. Save the fine-tuned model to the output directory.
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
