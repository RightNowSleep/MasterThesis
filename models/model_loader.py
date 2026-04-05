"""Model and tokenizer loading utilities for LLaMA with RoPE scaling support.

This module provides high-level functions for loading LLaMA models and tokenizers
with comprehensive support for various RoPE (Rotary Position Embedding) scaling
configurations, quantization settings, LoRA adapter merging, and gradient checkpointing.

Main Functions:
    - load_model: Load a LlamaForCausalLM model with specified RoPE configuration.
    - load_tokenizer: Load the tokenizer associated with a pretrained model.
    - add_args_model: Register CLI arguments for model configuration.

Key Features:
    - Automatic RoPE scaling configuration from command-line arguments
    - Support for 20+ different RoPE scaling types (linear, NTK-aware, YaRN, etc.)
    - Static vs dynamic scaling mode selection
    - BitsAndBytes quantization (4-bit and 8-bit) integration
    - LoRA adapter loading and merging via PEFT
    - Base adapter loading: Support for loading a foundational RoPE adapter before applying
      target scaling, enabling hierarchical RoPE method composition
    - Gradient checkpointing for memory-efficient training
    - Flexible dtype selection (float32, float16, bfloat16, auto)

Usage Example:
    Typical workflow involves calling add_args_model() to register arguments,
    parsing them, then passing the args namespace to load_model() and load_tokenizer().
"""

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from typing import Optional, Dict

from .configuration_llama import LlamaConfig
from .modeling_llama import LlamaForCausalLM


# ---------------------------------------------------------------------------
# RoPE type constants — mirrors the valid_types in LlamaConfig._rope_scaling_validation
# ---------------------------------------------------------------------------
_ROPE_TYPES_WITH_DYNAMIC_FLAG = {
    "linear",
    "ntk",
    "part-ntk",
    "yarn",
    "my-rope",
    "my-rope-scaled",
    "my-rope2",
    "my-rope2-scaled",
    "block-layered",
    "block-layered-scaled",
    "freq-smooth",
    "freq-smooth-scaled",
    "freq-reciprocal",
    "freq-reciprocal-scaled",
    "freq-reciprocal-scaled-no-layer",
    "freq-reciprocal-scaled-adaptive",
    "dual-rope",
    "dual-rope-scaled",
    "inverse-dual-rope",
    "inverse-dual-rope-scaled",
}
_ROPE_TYPE_NONE = "none"

_ALL_ROPE_TYPES = [_ROPE_TYPE_NONE] + sorted(_ROPE_TYPES_WITH_DYNAMIC_FLAG)


# ============================================================================ #
#  Internal helpers                                                            #
# ============================================================================ #


def _build_rope_scaling(args) -> Optional[Dict]:
    """
    Translate flat CLI arguments into the ``rope_scaling`` dict expected by ``LlamaConfig``.

    Rules:
        - ``--rope-type none``  → returns ``None``  (standard RoPE, no scaling)
        - ``--rope-factor`` and ``--rope-dynamic`` are **mutually exclusive** for all
          scaling types (linear / ntk / part-ntk / yarn / my-rope / my-rope2 /
          block-layered / block-layered-scaled / freq-smooth / freq-smooth-scaled):

          - ``--rope-factor F``   (F > 1.0)  — static scaling with a fixed ratio.
          - ``--rope-dynamic``               — dynamic scaling; ratio derived at
            runtime as ``s = max(1, L / L_orig)``.
          - Both supplied simultaneously     — ``--rope-factor`` wins and a warning
            is emitted; ``--rope-dynamic`` is ignored.
          - Neither supplied                 — raises ``ValueError``.

    Args:
        args: Parsed argument namespace containing rope_type, rope_factor, and rope_dynamic attributes.

    Returns:
        Optional[Dict]: A dictionary with 'type' and either 'factor' or 'dynamic' keys,
            or None if rope_type is 'none'.

    Raises:
        ValueError: If neither ``--rope-factor`` nor ``--rope-dynamic`` is provided.
    """
    if args.rope_type == _ROPE_TYPE_NONE:
        return None

    has_factor = args.rope_factor is not None
    has_dynamic = bool(args.rope_dynamic)

    if not (has_factor or has_dynamic):
        raise ValueError(
            f"--rope-type '{args.rope_type}' requires either --rope-factor (float > 1.0) "
            "or --rope-dynamic, but neither was provided."
        )

    if has_factor and has_dynamic:
        print(
            f"[WARNING] --rope-factor and --rope-dynamic are mutually exclusive. "
            f"--rope-factor {args.rope_factor} takes priority; --rope-dynamic ignored."
        )
        has_dynamic = False

    rope_scaling: dict = {"type": args.rope_type}

    if has_factor:
        rope_scaling["factor"] = float(args.rope_factor)
    else:
        rope_scaling["dynamic"] = True

    if hasattr(args, "rope_alpha") and args.rope_alpha is not None:
        rope_scaling["alpha"] = float(args.rope_alpha)
    if hasattr(args, "rope_beta") and args.rope_beta is not None:
        rope_scaling["beta"] = float(args.rope_beta)
    if hasattr(args, "rope_gamma") and args.rope_gamma is not None:
        rope_scaling["gamma"] = float(args.rope_gamma)

    return rope_scaling


def _resolve_torch_dtype(dtype_str: str):
    """
    Map a dtype string to a torch.dtype (or the literal string 'auto').

    Args:
        dtype_str: A string representing the data type. Valid values are 'float32',
            'float16', 'bfloat16', or 'auto'.

    Returns:
        Union[torch.dtype, str]: The corresponding torch.dtype or the string 'auto'.
    """
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": "auto",
    }
    return mapping[dtype_str]


# ============================================================================ #
#  Public API                                                                  #
# ============================================================================ #


def load_model(args, quantization_config=None):
    """
    Load a LlamaForCausalLM model with support for three modes:
    1. Base Adapter Mode (--base-adapter-path): Load base adapter + optional RoPE override + optional LoRA
    2. Adapter Mode (--adapter-path): Load complete trained model with LoRA weights
    3. Base Model Mode (default): Apply RoPE config directly to pretrained model

    Each branch is self-contained and returns independently.
    """
    print(f"Loading model : {args.model_name}")

    # ════════════════════════════════════════════════════════════
    # Branch 1: Base Adapter Mode
    # Load base RoPE adapter, optionally override its RoPE config,
    # then merge base adapter weights into the model.
    # ════════════════════════════════════════════════════════════
    if getattr(args, "base_adapter_path", None):
        print(f"  [Mode 1] Base Adapter: {args.base_adapter_path}")

        # Step 1: Load config from base adapter
        config = LlamaConfig.from_pretrained(
            args.base_adapter_path,
            trust_remote_code=True,
        )
        config.original_max_position_embeddings = getattr(
            config,
            "original_max_position_embeddings",
            config.max_position_embeddings,
        )

        # Step 2: Optionally override RoPE config via CLI args
        if args.rope_type != "none" or args.rope_factor is not None:
            rope_scaling = _build_rope_scaling(args)
            config.rope_scaling = rope_scaling
            config._rope_scaling_validation()
            print(
                f"  [INFO] Overriding base adapter RoPE with --rope-type={args.rope_type}"
            )

        # Step 3: Set runtime parameters
        config.max_position_embeddings = args.max_length
        config.use_cache = args.use_cache

        print(
            f"  rope-scaling   : {config.rope_scaling}\n"
            f"  max-length     : {config.max_position_embeddings}\n"
            f"  original_max   : {config.original_max_position_embeddings}"
        )

        # Step 4: Handle quantization config
        torch_dtype = _resolve_torch_dtype(args.dtype)
        if (args.load_in_8bit or args.load_in_4bit) and quantization_config is None:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit,
            )
            torch_dtype = "auto"
            print(f"  quantization: 4bit={args.load_in_4bit}, 8bit={args.load_in_8bit}")

        # Step 5: Load model weights (only once!)
        model = LlamaForCausalLM.from_pretrained(
            args.model_name,
            config=config,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config,
        )

        # Step 6: Merge base adapter LoRA weights
        print(f"  Merging base adapter: {args.base_adapter_path}")
        model = PeftModel.from_pretrained(model, args.base_adapter_path)
        model = model.merge_and_unload()
        print("  ✓ Base adapter merged successfully")

        # Step 7: Optionally merge additional LoRA adapter (e.g., fine-tuned on top)
        if getattr(args, "adapter_path", None):
            print(f"  Merging LoRA adapter: {args.adapter_path}")
            model = PeftModel.from_pretrained(model, args.adapter_path)
            model = model.merge_and_unload()
            print("  ✓ LoRA adapter merged successfully")

        # Step 8: Optional gradient checkpointing
        if getattr(args, "gradient_checkpointing", False):
            model.gradient_checkpointing_enable()
            print("  ✓ Gradient checkpointing enabled")

        print(f"✓ Model loaded (Mode 1: Base Adapter) | Device: {model.device}")
        return model, config

    # ════════════════════════════════════════════════════════════
    # Branch 2: Traditional Adapter Mode
    # Load complete trained model from adapter path (including RoPE config and LoRA weights).
    # Ignores --rope-type argument.
    # ════════════════════════════════════════════════════════════
    elif args.adapter_path:
        print(f"  [Mode 2] Traditional Adapter: {args.adapter_path}")

        # Step 1: Load complete config from adapter
        config = LlamaConfig.from_pretrained(
            args.adapter_path,
            trust_remote_code=True,
        )
        config.original_max_position_embeddings = getattr(
            config,
            "original_max_position_embeddings",
            config.max_position_embeddings,
        )

        # Warn user if --rope-type is also specified (will be ignored)
        if args.rope_type != "none" or args.rope_factor is not None:
            print(
                "[WARNING] --rope-type/--rope-factor is ignored when --adapter-path is set; "
                "RoPE config is loaded from the adapter's config.json."
            )

        # Step 2: Set runtime parameters
        config.max_position_embeddings = args.max_length
        config.use_cache = args.use_cache

        print(
            f"  rope-scaling   : {config.rope_scaling}\n"
            f"  max-length     : {config.max_position_embeddings}\n"
            f"  original_max   : {config.original_max_position_embeddings}"
        )

        # Step 3: Handle quantization config
        torch_dtype = _resolve_torch_dtype(args.dtype)
        if (args.load_in_8bit or args.load_in_4bit) and quantization_config is None:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit,
            )
            torch_dtype = "auto"
            print(f"  quantization: 4bit={args.load_in_4bit}, 8bit={args.load_in_8bit}")

        # Step 4: Load model weights
        model = LlamaForCausalLM.from_pretrained(
            args.model_name,
            config=config,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config,
        )

        # Step 5: Merge LoRA adapter weights
        print(f"  Loading LoRA adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()
        print("  ✓ LoRA adapter merged successfully")

        # Step 6: Optional gradient checkpointing
        if getattr(args, "gradient_checkpointing", False):
            model.gradient_checkpointing_enable()
            print("  ✓ Gradient checkpointing enabled")

        print(f"✓ Model loaded (Mode 2: Adapter) | Device: {model.device}")
        return model, config

    # ════════════════════════════════════════════════════════════
    # Branch 3: Base Model Mode (default)
    # Apply RoPE scaling config directly to pretrained model, no adapters used.
    # Used for quick experiments and prototyping new RoPE methods.
    # ════════════════════════════════════════════════════════════
    else:
        print(f"  [Mode 3] Base Model (no adapter)")

        # Step 1: Load default config from pretrained model
        config = LlamaConfig.from_pretrained(
            args.model_name,
            trust_remote_code=True,
        )
        config.original_max_position_embeddings = config.max_position_embeddings

        # Step 2: Build RoPE config from CLI args
        rope_scaling = _build_rope_scaling(args)
        config.rope_scaling = rope_scaling
        config._rope_scaling_validation()

        # Step 3: Set runtime parameters
        config.max_position_embeddings = args.max_length
        config.use_cache = args.use_cache

        print(
            f"  rope-scaling   : {config.rope_scaling}\n"
            f"  max-length     : {config.max_position_embeddings}\n"
            f"  original_max   : {config.original_max_position_embeddings}"
        )

        # Step 4: Handle quantization config
        torch_dtype = _resolve_torch_dtype(args.dtype)
        if (args.load_in_8bit or args.load_in_4bit) and quantization_config is None:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit,
            )
            torch_dtype = "auto"
            print(f"  quantization: 4bit={args.load_in_4bit}, 8bit={args.load_in_8bit}")

        # Step 5: Load model weights
        model = LlamaForCausalLM.from_pretrained(
            args.model_name,
            config=config,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config,
        )

        # Step 6: Optional gradient checkpointing
        if getattr(args, "gradient_checkpointing", False):
            model.gradient_checkpointing_enable()
            print("  ✓ Gradient checkpointing enabled")

        print(f"✓ Model loaded (Mode 3: Base Model) | Device: {model.device}")
        return model, config


def load_tokenizer(args):
    """
    Load the tokenizer associated with ``args.model_name``.

    Args:
        args: Parsed argument namespace (see ``add_args_model``).

    Returns:
        PreTrainedTokenizer: The loaded tokenizer with pad_token set to eos_token
            if not already defined.
    """
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = args.max_length
    print(f"Tokenizer loaded  |  vocab_size: {tokenizer.vocab_size}")
    return tokenizer


# ============================================================================ #
#  Argument definitions                                                        #
# ============================================================================ #


def add_args_model(parser):
    """
    Register all model-loading arguments with ``parser``.

    Argument groups:
        - Model identity: --model-name, --adapter-path
        - Hardware: --device, --dtype, --load-in-8bit, --load-in-4bit
        - Training helpers: --use-cache, --gradient-checkpointing
        - Sequence length: --max-length, --min-length
        - RoPE scaling: --rope-type, --rope-factor, --rope-dynamic

    Args:
        parser: argparse.ArgumentParser instance to add arguments to.

    Returns:
        argparse.ArgumentParser: The same parser with arguments added (for chaining).
    """
    # ── Model identity ──────────────────────────────────────────────── #
    parser.add_argument(
        "--model-name",
        type=str,
        default="huggyllama/llama-7b",
        help="HuggingFace model identifier or local path to pretrained weights.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Local path to a LoRA adapter directory.  "
        "If provided, the adapter is loaded via PEFT and merged into the "
        "base model before returning. Remember not to checkpoint the adapter.",
    )
    parser.add_argument(
        "--base-adapter-path",
        type=str,
        default=None,
        help="Local path to a base adapter directory containing the foundational RoPE method. "
        "If provided, this adapter's weights and RoPE config are loaded first, then "
        "the target --rope-type (if specified) is applied on top. This enables "
        "training scaled variants (e.g., inverse-dual-rope-scaled) on top of their "
        "base methods (e.g., inverse-dual-rope). The base adapter is merged into "
        "the model before any optional --adapter-path LoRA adapter.",
    )

    # ── Hardware ────────────────────────────────────────────────────── #
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Visible CUDA device indices, e.g. '0,1,2'.  "
        "Sets CUDA_VISIBLE_DEVICES when passed to the launch script.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["float32", "float16", "bfloat16", "auto"],
        help="Floating-point precision for model weights.  "
        "'auto' lets HuggingFace infer from the checkpoint.",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit via bitsandbytes.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit (QLoRA-style) via bitsandbytes.",
    )

    # ── Training helpers ────────────────────────────────────────────── #
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Enable the KV cache (config.use_cache=True).  "
        "Should be disabled during gradient-checkpoint training.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to trade compute for memory.",
    )

    # ── Sequence length ─────────────────────────────────────────────── #
    parser.add_argument(
        "--max-length",
        type=int,
        default=8 * 1024,
        help="Maximum token sequence length the model will be asked to process.  "
        "Sets config.max_position_embeddings.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=2 * 1024,
        help="Minimum token sequence length (used by downstream evaluation scripts).",
    )

    # ── RoPE scaling ────────────────────────────────────────────────── #
    parser.add_argument(
        "--rope-type",
        type=str,
        default=_ROPE_TYPE_NONE,
        choices=_ALL_ROPE_TYPES,
        help=(
            "Context-extension strategy for RoPE positional embeddings.\n"
            "  none                  — standard RoPE, no scaling\n"
            "  linear                — Position Interpolation (PI)\n"
            "  ntk                   — NTK-aware base scaling\n"
            "  part-ntk              — NTK-by-parts (per-dimension blending)\n"
            "  yarn                  — YaRN (NTK-by-parts + attention temperature)\n"
            "  my-rope               — layer-aware RoPE (position only)\n"
            "  my-rope-scaled        — layer-aware RoPE + attention temperature\n"
            "  my-rope2              — multi-scale subspace RoPE (position only)\n"
            "  my-rope2-scaled       — multi-scale subspace RoPE + attention temperature\n"
            "  block-layered         — Block-Layered RoPE (position only)\n"
            "  block-layered-scaled  — Block-Layered RoPE + attention temperature\n"
            "  freq-smooth           — Freq-Smooth RoPE (position only)\n"
            "  freq-smooth-scaled    — Freq-Smooth RoPE + attention temperature\n"
            "  freq-reciprocal       — Freq-Reciprocal RoPE (position only)\n"
            "  freq-reciprocal-scaled — Freq-Reciprocal RoPE + attention temperature\n"
            "  freq-reciprocal-scaled-no-layer — Freq-Reciprocal RoPE + attention temperature, no layer index\n"
            "  freq-reciprocal-scaled-adaptive — Freq-Reciprocal RoPE + adaptive attention temperature\n"
            "  dual-rope               — Dual RoPE (position only)\n"
            "  dual-rope-scaled        — Dual RoPE + attention temperature\n"
            "  inverse-dual-rope          — Inverse-Dual RoPE (position only)\n"
            "  inverse-dual-rope-scaled    — Inverse-Dual RoPE + global-local attention scaling (alpha/beta/gamma)\n"
            "\n"
            "All types except 'none' require exactly ONE of:\n"
            "  --rope-factor F   static scaling with fixed ratio F > 1.0\n"
            "  --rope-dynamic    dynamic scaling; ratio derived at runtime\n"
            "If both are given, --rope-factor wins and --rope-dynamic is ignored."
        ),
    )
    parser.add_argument(
        "--rope-factor",
        type=float,
        default=None,
        help=(
            "Static context extension ratio s > 1.0.  Mutually exclusive with "
            "--rope-dynamic; if both are supplied, --rope-factor wins.  "
            "Example: --rope-factor 4.0 extends the context 4× beyond the "
            "model's original training length."
        ),
    )
    parser.add_argument(
        "--rope-dynamic",
        action="store_true",
        help=(
            "Enable dynamic scaling for all six RoPE types.  Mutually exclusive "
            "with --rope-factor; if both are supplied, --rope-factor wins.  "
            "In dynamic mode the scaling factor is computed on every forward pass "
            "as s = max(1, seq_len / original_L), so the model adapts automatically "
            "to any sequence length without reloading weights."
        ),
    )
    parser.add_argument(
        "--rope-alpha",
        type=float,
        default=0.06,
        help="Global term growth rate for inverse-dual-rope-scaled (alpha). Default: 0.1",
    )
    parser.add_argument(
        "--rope-beta",
        type=float,
        default=0.50,
        help="Boundary jump compensation amplitude for inverse-dual-rope-scaled (beta). Default: 0.5",
    )
    parser.add_argument(
        "--rope-gamma",
        type=float,
        default=4.05,
        help="Intra-segment decay rate for inverse-dual-rope-scaled (gamma). Default: 2.0",
    )

    return parser
