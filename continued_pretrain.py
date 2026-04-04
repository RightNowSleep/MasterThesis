"""Continued pretraining script for large language models with LoRA/QLoRA.

This module implements a full continued pretraining pipeline that extends the context
window of transformer-based language models using Low-Rank Adaptation (LoRA) or
Quantized LoRA (QLoRA). It supports:

    - Progressive length training: Gradually increases sequence length across stages
      to help the model adapt to longer contexts (e.g., [2048, 4096, 8192, 16384]).
    - Multiple RoPE scaling types: linear, NTK-aware, YaRN, and dynamic scaling.
    - Distributed training via Hugging Face Accelerate with mixed precision.
    - Checkpoint management with automatic rotation and resume capability.
    - Integration with Weights & Biases for experiment tracking.

The training loop follows a standard pretraining paradigm: load model -> wrap with
LoRA -> iterate over dataset with gradient accumulation -> save checkpoints at
regular intervals.

Typical usage::

    python continued_pretrain.py \\
        --model-name meta-llama/Llama-2-7b-hf \\
        --rope-type linear --rope-factor 4.0 \\
        --max-length 16384 --progressive-length \\
        --quantization 4bit --lora-r 64
"""

import argparse
import torch
import os
import shutil
import json
import time
import warnings
import wandb
from datetime import timedelta

from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed
from tqdm import tqdm
from transformers import (
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import bitsandbytes as bnb

from models.model_loader import load_model, load_tokenizer, add_args_model

warnings.filterwarnings("ignore")


def find_all_linear_names(model):
    """Find all linear layer names in the model for LoRA targeting.

    Inspects all modules in the model and collects unique final layer name
    components that correspond to linear layers (including quantized variants).

    Args:
        model: The PyTorch model to inspect for linear layers.

    Returns:
        list[str]: List of unique linear layer names suitable for LoRA targeting,
            excluding ``lm_head`` which is typically not adapted.
    """
    linear_cls = (torch.nn.Linear,)
    try:
        linear_cls = linear_cls + (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    except AttributeError:
        pass

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_cls):
            lora_module_names.add(name.split(".")[-1])

    lora_module_names.discard("lm_head")
    return list(lora_module_names)


def make_collate_fn(max_length: int):
    """Create a collate function for batching dataset examples.

    Produces a closure that truncates sequences to *max_length*, stacks them
    into tensors, and returns a batch dictionary.

    Args:
        max_length: Maximum sequence length for truncation.

    Returns:
        Callable: A collate function that accepts a list of example dictionaries
            and returns a single batched dictionary with stacked tensor values.
    """

    def collate_fn(examples):
        batch = {}
        for key in examples[0]:
            vals = []
            for ex in examples:
                v = ex[key]
                v = v[:max_length]
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v, dtype=torch.long)
                vals.append(v)
            batch[key] = torch.stack(vals)
        return batch

    return collate_fn


def get_optimizer_param_groups(model, weight_decay: float):
    """Create parameter groups for optimizer with selective weight decay.

    Splits model parameters into two groups: those that should receive weight
    decay (e.g., attention/dense weights) and those that should not (e.g.,
    biases, layer norms, embeddings).

    Args:
        model: The model whose parameters will be grouped.
        weight_decay: Weight decay coefficient for applicable parameters.

    Returns:
        list[dict]: List of parameter group dictionaries, each containing
            ``params`` and ``weight_decay`` keys.
    """
    no_decay_keywords = [
        "bias",
        "layer_norm",
        "layernorm",
        "norm",
        "embed",
        "embed_tokens",
    ]
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(kw in name.lower() for kw in no_decay_keywords):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def save_checkpoint(
    accelerator,
    model,
    model_config,
    output_dir: str,
    step: int,
    epoch: int,
    max_checkpoints: int,
    args=None,
    tokenizer=None,
    stage_idx=None,
    stage_length=None,
):
    """Save a training checkpoint with model state and training configuration.

    Persists the LoRA adapter weights, accelerator state, model config,
    tokenizer (if provided), and training arguments. Automatically rotates
    old checkpoints when the count exceeds *max_checkpoints*.

    Args:
        accelerator: The Accelerator instance managing distributed training.
        model: The model to save (will be unwrapped before saving).
        model_config: The model configuration object to persist alongside weights.
        output_dir: Directory path for saving checkpoints.
        step: Current training step number (used in checkpoint directory name).
        epoch: Current epoch number.
        max_checkpoints: Maximum number of checkpoints to retain; older ones
            are deleted automatically.
        args: Optional training arguments namespace whose hyperparameters are
            serialized into ``training_args.pt``.
        tokenizer: Optional tokenizer to save alongside the model.
        stage_idx: Optional current progressive stage index for multi-stage training.
        stage_length: Optional current stage's max sequence length.

    Returns:
        None
    """
    checkpoint_path = os.path.join(output_dir, f"checkpoint_step_{step:06d}")
    os.makedirs(checkpoint_path, exist_ok=True)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.print(f"Saving checkpoint at step {step} (epoch {epoch})...")

    accelerator.save_state(os.path.join(checkpoint_path, "accelerator_state"))

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        os.path.join(checkpoint_path, "adapter_model"),
        safe_serialization=True,
    )

    model_config.save_pretrained(checkpoint_path)

    if tokenizer is not None:
        tokenizer.save_pretrained(checkpoint_path)

    if args is not None:
        training_state = {
            "checkpoint_step": step,
            "checkpoint_epoch": epoch,
            "learning_rate": args.learning_rate,
            "max_train_steps": args.max_train_steps,
            "warmup_steps": args.warmup_steps,
            "gradient_accumulate_every": args.gradient_accumulate_every,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "quantization": args.quantization,
            "lr_schedule": args.lr_schedule,
            "dtype": args.dtype,
            "rope_type": args.rope_type,
            "rope_factor": args.rope_factor,
            "rope_dynamic": args.rope_dynamic,
            "progressive_length": args.progressive_length,
            "current_stage": stage_idx,
            "stage_length": stage_length,
        }
        torch.save(training_state, os.path.join(checkpoint_path, "training_args.pt"))

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        all_ckpts = sorted(
            [d for d in os.listdir(output_dir) if d.startswith("checkpoint_step_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        while len(all_ckpts) > max_checkpoints:
            old = all_ckpts.pop(0)
            shutil.rmtree(os.path.join(output_dir, old))
            accelerator.print(f"Removed old checkpoint: {old}")
        accelerator.print(f"Checkpoint saved → {checkpoint_path}")


def generate_progressive_lengths(max_length: int, original_length: int):
    """Generate a progressive length schedule for staged context extension.

    Starting from *original_length*, each subsequent stage doubles the sequence
    length until reaching or exceeding *max_length*. The last value is clamped
    to exactly *max_length* to ensure the target is always included.

    For example, with ``max_length=16384`` and ``original_length=2048``, this
    produces ``[2048, 4096, 8192, 16384]``.

    Args:
        max_length: Target maximum sequence length to reach.
        original_length: Original model context window size (starting point).

    Returns:
        list[int]: Monotonically increasing list of sequence lengths for each
            training stage.
    """
    lengths = []
    current = original_length
    while current <= max_length:
        lengths.append(current)
        current *= 2
    # Ensure the last value equals max_length (when it is not an exact power-of-2 multiple)
    if lengths and lengths[-1] != max_length:
        lengths[-1] = max_length
    return lengths


def main(args):
    """Execute the continued pretraining pipeline with LoRA/QLoRA.

    Orchestrates the entire training workflow including environment setup,
    model loading with optional quantization, LoRA wrapping, dataset preparation,
    optimizer/scheduler construction, the progressive-length training loop,
    periodic checkpointing, and final model serialization.

    Args:
        args: Parsed arguments containing all training configuration including
            model settings, optimization parameters, LoRA configuration,
            quantization options, and output settings.

    Returns:
        None
    """
    # ------------------------------------------------------------------
    # Derive experiment tag from RoPE type
    # ------------------------------------------------------------------
    _tag = args.rope_type

    os.makedirs(args.output_dir, exist_ok=True)
    save_name = f"{_tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    model_dir = os.path.join(args.output_dir, save_name)
    os.makedirs(model_dir, exist_ok=True)

    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_loss_path = os.path.join(model_dir, "loss.csv")

    with open(os.path.join(model_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    if args.wandb and args.wandb.strip():
        wandb.login()

    set_seed(args.seed)

    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))

    if args.dtype == "bfloat16":
        mixed_precision = "bf16"
        torch_dtype = torch.bfloat16
    elif args.dtype == "float16":
        mixed_precision = "fp16"
        torch_dtype = torch.float16
    else:
        mixed_precision = None
        torch_dtype = "auto"

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulate_every,
        mixed_precision=mixed_precision,
        log_with="wandb" if args.wandb else None,
        kwargs_handlers=[timeout],
    )
    accelerator.init_trackers(
        project_name=args.wandb if args.wandb else "continued_pretrain",
        config=vars(args),
    )
    accelerator.print(f"Total GPUs : {accelerator.num_processes}")
    accelerator.print(f"Visible    : {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")

    # ------------------------------------------------------------------
    # Build QLoRA-grade quantization config if requested
    # ------------------------------------------------------------------
    quantization_config = None
    if args.quantization in ("4bit", "8bit"):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=(args.quantization == "4bit"),
            load_in_8bit=(args.quantization == "8bit"),
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        accelerator.print(f"Quantization: {args.quantization} (QLoRA)")
    else:
        accelerator.print(f"Quantization: none, dtype={args.dtype}")

    # ------------------------------------------------------------------
    # Patch args so load_model behaves correctly for training:
    #   - use_cache=False          – required during training
    #   - gradient_checkpointing=False – handled manually below
    #   - adapter_path=None        – training creates LoRA, not loads one
    #   - load_in_4bit / 8bit      – aligned with --quantization
    # ------------------------------------------------------------------
    args.use_cache = False
    args.gradient_checkpointing = False  # handled below, after LoRA wrapping
    args.adapter_path = None
    args.load_in_4bit = args.quantization == "4bit"
    args.load_in_8bit = args.quantization == "8bit"

    # ------------------------------------------------------------------
    # Load model via unified loader.
    # load_model returns (model, config); config is stored for later saving.
    # ------------------------------------------------------------------
    accelerator.print("Loading model...")
    model, model_config = load_model(args, quantization_config=quantization_config)

    # Save config snapshot
    if hasattr(model_config, "save_pretrained"):
        model_config.save_pretrained(model_dir)

    # ------------------------------------------------------------------
    # Progressive length training setup
    # ------------------------------------------------------------------
    if args.progressive_length:
        original_length = model_config.original_max_position_embeddings
        progressive_lengths = generate_progressive_lengths(
            args.max_length, original_length
        )
        accelerator.print(f"\nProgressive length training enabled!")
        accelerator.print(f"Length stages: {progressive_lengths}")
        accelerator.print(f"Total stages: {len(progressive_lengths)}\n")
    else:
        progressive_lengths = [args.max_length]

    # ------------------------------------------------------------------
    # Gradient checkpointing (must happen before LoRA wrapping)
    # ------------------------------------------------------------------
    if args.quantization in ("4bit", "8bit"):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()

    model.enable_input_require_grads()

    # ------------------------------------------------------------------
    # LoRA wrapping
    # ------------------------------------------------------------------
    target_modules = find_all_linear_names(model)
    accelerator.print(f"LoRA target modules: {target_modules}")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Dataset & DataLoader
    # ------------------------------------------------------------------
    accelerator.print("Loading dataset...")
    train_dataset = load_dataset(args.dataset, split="train")
    collate_fn = make_collate_fn(args.max_length)

    num_workers = min(4, max(os.cpu_count() // max(accelerator.num_processes, 1), 1))
    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    param_groups = get_optimizer_param_groups(model, args.weight_decay)

    if args.quantization in ("4bit", "8bit"):
        accelerator.print("Optimizer: AdamW8bit (memory-efficient, QLoRA)")
        optim = bnb.optim.AdamW8bit(param_groups, lr=args.learning_rate)
    else:
        accelerator.print("Optimizer: AdamW")
        optim = torch.optim.AdamW(param_groups, lr=args.learning_rate)

    # ------------------------------------------------------------------
    # LR scheduler
    # ------------------------------------------------------------------
    base_kwargs = dict(optimizer=optim, num_warmup_steps=args.warmup_steps)
    if args.lr_schedule == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            **base_kwargs,
            num_training_steps=args.max_train_steps,
        )
    elif args.lr_schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(
            **base_kwargs,
            num_training_steps=args.max_train_steps,
        )
    else:
        scheduler = get_constant_schedule_with_warmup(**base_kwargs)

    optim, train_loader, scheduler = accelerator.prepare(optim, train_loader, scheduler)
    accelerator.register_for_checkpointing(scheduler)

    total_batch_size = (
        args.batch_size * accelerator.num_processes * args.gradient_accumulate_every
    )
    accelerator.print(f"Max train steps      : {args.max_train_steps}")
    accelerator.print(f"Warmup steps         : {args.warmup_steps}")
    accelerator.print(f"Total batch size     : {total_batch_size}")
    accelerator.print(f"Max length           : {args.max_length}")
    accelerator.print(f"RoPE type            : {args.rope_type}")
    accelerator.print(f"RoPE factor          : {args.rope_factor}")
    accelerator.print(f"RoPE dynamic         : {args.rope_dynamic}")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tokenizer = load_tokenizer(args)

    # ------------------------------------------------------------------
    # Optional checkpoint resume
    # ------------------------------------------------------------------
    completed_steps = 0
    start_epoch = 0
    resume_batch_skip = 0

    if args.resume_from_checkpoint:
        resume_path = args.resume_from_checkpoint
        accelerator.print(f"Resuming from: {resume_path}")

        state_path = os.path.join(resume_path, "training_args.pt")
        if os.path.exists(state_path):
            saved = torch.load(state_path, map_location="cpu")
            completed_steps = saved.get("checkpoint_step", 0)
            start_epoch = saved.get("checkpoint_epoch", 0)
            accelerator.print(f"  → step={completed_steps}, epoch={start_epoch}")

        accelerator.load_state(os.path.join(resume_path, "accelerator_state"))

        batches_per_epoch = len(train_loader)
        total_done_batches = completed_steps * args.gradient_accumulate_every
        resume_batch_skip = total_done_batches - start_epoch * batches_per_epoch
        resume_batch_skip = max(0, resume_batch_skip)
        accelerator.print(
            f"  → skipping {resume_batch_skip} batches in epoch {start_epoch}"
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=completed_steps,
        disable=not accelerator.is_local_main_process,
    )

    loss_file = open(log_loss_path, "a" if args.resume_from_checkpoint else "w")
    if not args.resume_from_checkpoint:
        loss_file.write("timestamp,step,loss,lr\n")

    if args.save_only:
        accelerator.print("save_only=True, skipping training.")
    else:
        model.train()
        epoch = start_epoch

        for stage_idx, stage_length in enumerate(progressive_lengths):
            current_max_length = stage_length
            steps_per_stage = args.max_train_steps // len(progressive_lengths)

            accelerator.print(f"\n{'='*60}")
            accelerator.print(
                f"Stage {stage_idx + 1}/{len(progressive_lengths)}: "
                f"training with max_length={stage_length}"
            )
            accelerator.print(f"Steps for this stage: {steps_per_stage}")
            accelerator.print(f"{'='*60}\n")

            collate_fn = make_collate_fn(stage_length)

            num_workers = min(
                4, max(os.cpu_count() // max(accelerator.num_processes, 1), 1)
            )
            train_loader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                shuffle=True,
                batch_size=args.batch_size,
                pin_memory=True,
                num_workers=num_workers,
                drop_last=True,
            )
            train_loader = accelerator.prepare(train_loader)

            stage_completed = 0

            while (
                stage_completed < steps_per_stage
                and completed_steps < args.max_train_steps
            ):
                set_seed(args.seed + epoch)

                if resume_batch_skip > 0:
                    data_iter = accelerator.skip_first_batches(
                        train_loader,
                        resume_batch_skip,
                    )
                    resume_batch_skip = 0
                else:
                    data_iter = train_loader

                for batch in data_iter:
                    with accelerator.accumulate(model):
                        outputs = model(**batch)
                        loss = outputs.loss
                        accelerator.backward(loss)

                        if accelerator.sync_gradients:
                            avg_loss = accelerator.gather(loss.detach()).mean().item()
                            if args.grad_norm is not None and args.grad_norm > 0:
                                accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.grad_norm,
                                )

                        optim.step()
                        scheduler.step()
                        optim.zero_grad()

                    if accelerator.sync_gradients:
                        completed_steps += 1
                        stage_completed += 1
                        current_lr = scheduler.get_last_lr()[0]
                        log_data = {"loss": avg_loss, "lr": current_lr}

                        progress_bar.update(1)
                        progress_bar.set_postfix(
                            {
                                **log_data,
                                "stage": f"{stage_idx+1}/{len(progressive_lengths)}",
                                "length": stage_length,
                            }
                        )
                        accelerator.log(log_data, step=completed_steps)

                        if accelerator.is_main_process:
                            loss_file.write(
                                f"{time.time()},{completed_steps},{avg_loss},{current_lr}\n"
                            )
                            loss_file.flush()

                        if (
                            completed_steps > 0
                            and completed_steps % args.checkpointing_steps == 0
                        ):
                            save_checkpoint(
                                accelerator=accelerator,
                                model=model,
                                model_config=model_config,
                                output_dir=checkpoint_dir,
                                step=completed_steps,
                                epoch=epoch,
                                max_checkpoints=args.max_checkpoints,
                                args=args,
                                tokenizer=tokenizer,
                                stage_idx=(
                                    stage_idx if args.progressive_length else None
                                ),
                                stage_length=(
                                    stage_length if args.progressive_length else None
                                ),
                            )

                        if (
                            completed_steps >= args.max_train_steps
                            or stage_completed >= steps_per_stage
                        ):
                            break

                epoch += 1

        accelerator.print("Training finished.")
        accelerator.end_training()

    loss_file.close()

    # ------------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------------
    accelerator.wait_for_everyone()
    accelerator.print(f"Saving final model → {model_dir}")

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)

    accelerator.print("Done.")


def add_args_continued_pretrain(parser):
    """Add continued pretraining specific arguments to the argument parser.

    Registers all CLI flags related to dataset selection, optimization
    hyperparameters, LoRA configuration, quantization mode, checkpointing
    strategy, and infrastructure options (WandB, progressive length, etc.).

    Args:
        parser: The argparse.ArgumentParser instance to add arguments to.

    Returns:
        argparse.ArgumentParser: The modified argument parser with continued
            pretraining arguments registered.
    """
    # ── Dataset ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--dataset",
        type=str,
        default="emozilla/pg_books-tokenized-bos-eos-chunked-65536",
        help="HuggingFace dataset id for continued pretraining.",
    )

    # ── Optimisation ─────────────────────────────────────────────────────────
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-steps", type=int, default=600)
    parser.add_argument("--warmup-steps", type=int, default=60)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-GPU batch size.",
    )
    parser.add_argument("--gradient-accumulate-every", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--grad-norm",
        type=float,
        default=1.0,
        help="Max gradient norm (0 = disable).",
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="cosine",
        choices=["linear", "constant", "cosine"],
    )

    # ── LoRA ─────────────────────────────────────────────────────────────────
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=128,
        help="LoRA alpha. Recommend >= lora_r.",
    )
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # ── Quantization (QLoRA) ─────────────────────────────────────────────────
    parser.add_argument(
        "--quantization",
        type=str,
        default="4bit",
        choices=["4bit", "8bit", "none"],
        help="Quantization mode for QLoRA training. "
        "The matching load_in_4bit/load_in_8bit flags are set automatically.",
    )

    # ── Checkpointing ────────────────────────────────────────────────────────
    parser.add_argument("--checkpointing-steps", type=int, default=120)
    parser.add_argument("--max-checkpoints", type=int, default=2)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint_step_XXXXXX directory to resume from.",
    )

    # ── Output ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output-dir",
        type=str,
        default="finetunes/continued_pretrain",
        help="Root directory for saving models and checkpoints.",
    )

    # ── Infrastructure ───────────────────────────────────────────────────────
    parser.add_argument(
        "--wandb",
        type=str,
        default=None,
        help="WandB project name. Leave empty to disable.",
    )
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Save model without training (dry-run).",
    )
    parser.add_argument(
        "--progressive-length",
        action="store_true",
        help=(
            "Enable progressive length training for context extension. "
            "When enabled, training progresses through increasing sequence lengths "
            "(e.g., [2048, 4096, 8192, 16384]) instead of jumping directly to max_length. "
            "This helps the model gradually adapt to longer contexts."
        ),
    )

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Continued Pretraining with LoRA/QLoRA"
    )
    parser = add_args_model(parser)
    parser = add_args_continued_pretrain(parser)
    main(parser.parse_args())
