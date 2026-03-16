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
    output_dir: str,
    step: int,
    epoch: int,
    max_checkpoints: int,
    args=None,
    tokenizer=None,
):
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


def main(args):
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # ------------------------------------------------------------------
    # Derive experiment tag
    # ------------------------------------------------------------------
    _tag = args.rope_type

    os.makedirs(args.output_dir, exist_ok=True)
    save_name = f"{_tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    model_dir = os.path.join(args.output_dir, save_name)
    os.makedirs(model_dir, exist_ok=True)

    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_loss_path = os.path.join(model_dir, "loss.csv")

    with open(os.path.join(model_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

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
    # Build QLoRA-grade quantization config if requested.
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
    #   • use_cache=False          – required during training
    #   • gradient_checkpointing=False – handled manually below
    #   • adapter_path=None        – training creates LoRA, not loads one
    #   • load_in_4bit / 8bit      – aligned with --quantization
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

        while completed_steps < args.max_train_steps:
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
                    current_lr = scheduler.get_last_lr()[0]
                    log_data = {"loss": avg_loss, "lr": current_lr}

                    progress_bar.update(1)
                    progress_bar.set_postfix(log_data)
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
                            output_dir=checkpoint_dir,
                            step=completed_steps,
                            epoch=epoch,
                            max_checkpoints=args.max_checkpoints,
                            args=args,
                            tokenizer=tokenizer,
                        )

                    if completed_steps >= args.max_train_steps:
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
    parser.add_argument("--batch-size", type=int, default=1, help="Per-GPU batch size.")
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
        default="/home/linzhen/workspace/finetunes/continued_pretrain",
        help="Root directory for saving models and checkpoints.",
    )

    # ── Infrastructure ───────────────────────────────────────────────────────
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default="1,2,3",
        help="Comma-separated CUDA device ids to expose.",
    )
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

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Continued Pretraining with LoRA/QLoRA"
    )
    parser = add_args_model(parser)
    parser = add_args_continued_pretrain(parser)
    main(parser.parse_args())
