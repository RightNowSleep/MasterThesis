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
}
_ROPE_TYPE_NONE = "none"

_ALL_ROPE_TYPES = [_ROPE_TYPE_NONE] + sorted(_ROPE_TYPES_WITH_DYNAMIC_FLAG)


# ============================================================================ #
#  Internal helpers                                                            #
# ============================================================================ #


def _build_rope_scaling(args) -> Optional[Dict]:
    r"""
    Translate flat CLI arguments into the ``rope_scaling`` dict expected by
    ``LlamaConfig``.

    Rules
    -----
    * ``--rope-type none``  → returns ``None``  (standard RoPE, no scaling)
    * ``--rope-factor`` and ``--rope-dynamic`` are **mutually exclusive** for all
      six scaling types (linear / ntk / part-ntk / yarn / my-rope / my-rope2 /
      block-layered / block-layered-scaled / freq-smooth / freq-smooth-scaled):

      - ``--rope-factor F``   (F > 1.0)  — static scaling with a fixed ratio.
      - ``--rope-dynamic``               — dynamic scaling; ratio derived at
                                           runtime as ``s = max(1, L / L_orig)``.
      - Both supplied simultaneously     — ``--rope-factor`` wins and a warning
                                           is emitted; ``--rope-dynamic`` is
                                           ignored.
      - Neither supplied                 — raises ``ValueError``.

    Raises
    ------
    ValueError
        If neither ``--rope-factor`` nor ``--rope-dynamic`` is provided.
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

    return rope_scaling


def _resolve_torch_dtype(dtype_str: str):
    """Map a dtype string to a torch.dtype (or the literal string 'auto')."""
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
    Load a LlamaForCausalLM model from pretrained weights with the RoPE
    configuration specified by ``args``.

    Parameters
    ----------
    args :
        Parsed argument namespace (see ``add_args_model``).
    quantization_config : BitsAndBytesConfig, optional
        If provided, overrides any quantization settings derived from args.

    Returns
    -------
    model : LlamaForCausalLM
        Loaded (and optionally adapter-merged) model.
    config : LlamaConfig
        Final model configuration used for loading.
    """
    print(f"Loading model : {args.model_name}")
    if args.adapter_path:
        print(f"  adapter: {args.adapter_path}")
        config = LlamaConfig.from_pretrained(
            args.adapter_path,
            trust_remote_code=True,
        )
        config.original_max_position_embeddings = getattr(
            config,
            "original_max_position_embeddings",
            config.max_position_embeddings,
        )
        if args.rope_type != "none" or args.rope_factor is not None:
            print(
                "[WARNING] --rope-type/--rope-factor are ignored when --adapter-path is set;"
                " RoPE config is loaded from the adapter's config.json."
            )
    else:
        config = LlamaConfig.from_pretrained(
            args.model_name,
            trust_remote_code=True,
        )
        config.original_max_position_embeddings = config.max_position_embeddings
        rope_scaling = _build_rope_scaling(args)
        config.rope_scaling = rope_scaling
        config._rope_scaling_validation()
    config.max_position_embeddings = args.max_length
    config.use_cache = args.use_cache

    print(
        f"  rope-scaling   : {config.rope_scaling}\n"
        f"  max-length  : {config.max_position_embeddings}\n"
        f"  original_max_position_embeddings: {config.original_max_position_embeddings}"
    )

    torch_dtype = _resolve_torch_dtype(args.dtype)

    if (args.load_in_8bit or args.load_in_4bit) and quantization_config is None:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
        )
        torch_dtype = "auto"
        print(f"  quantization: 4bit={args.load_in_4bit}, 8bit={args.load_in_8bit}")

    # ------------------------------------------------------------------ #
    # 6. Load model weights                                              #
    # ------------------------------------------------------------------ #
    model = LlamaForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
    )

    # ------------------------------------------------------------------ #
    # 7. Optional: merge LoRA adapter                                    #
    # ------------------------------------------------------------------ #
    if args.adapter_path:
        print(f"  Loading LoRA adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()
        print("  LoRA adapter merged successfully")

    # ------------------------------------------------------------------ #
    # 8. Optional: gradient checkpointing                                #
    # ------------------------------------------------------------------ #
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled")

    print(f"Model loaded successfully  |  device: {model.device}")
    return model, config


def load_tokenizer(args):
    """
    Load the tokenizer associated with ``args.model_name``.

    Parameters
    ----------
    args :
        Parsed argument namespace (see ``add_args_model``).

    Returns
    -------
    tokenizer : PreTrainedTokenizer
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

    Argument groups
    ---------------
    Model identity   : --model-name, --adapter-path
    Hardware         : --device, --dtype, --load-in-8bit, --load-in-4bit
    Training helpers : --use-cache, --gradient-checkpointing
    Sequence length  : --max-length, --min-length
    RoPE scaling     : --rope-type, --rope-factor, --rope-dynamic

    Parameters
    ----------
    parser : argparse.ArgumentParser

    Returns
    -------
    argparse.ArgumentParser
        The same parser with arguments added (for chaining).
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

    return parser
