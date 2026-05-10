"""Custom GGUF extensions for exporting LLaMA models with non-standard RoPE methods.

This module provides utilities to:
    1. Bake pre-computed RoPE cos/sin caches into exportable tensors (static mode).
    2. Extract and serialize custom RoPE metadata required to reconstruct
       arbitrary RoPE configurations at runtime.
    3. Write/read custom key-value pairs into/from GGUF files using the
       official ``gguf`` package.

The design keeps llama.cpp C++ code untouched: standard GGUF conversion
handles the weight matrices, while this module injects extra metadata and
(optionally) baked cache tensors that a Python backend can consume later.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from .configuration_llama import LlamaConfig


# ---------------------------------------------------------------------------
# RoPE type classification
# ---------------------------------------------------------------------------

TIER1_ROPE_TYPES = {
    "none",
    "linear",
    "ntk",
    "part-ntk",
    "yarn",
}

TIER2_ROPE_TYPES = {
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
}

TIER3_ROPE_TYPES = {
    "inverse-dual-rope",
    "inverse-dual-rope-scaled",
    "inverse-dual-tangle-rope",
    "inverse-dual-tangle-rope-scaled",
    "inverse-dual-nopos-rope",
    "inverse-dual-nopos-rope-scaled",
}

ALL_CUSTOM_ROPE_TYPES = TIER2_ROPE_TYPES | TIER3_ROPE_TYPES


def _is_tier1(rope_type: Optional[str]) -> bool:
    return rope_type is None or rope_type in TIER1_ROPE_TYPES


def _is_custom(rope_type: Optional[str]) -> bool:
    return rope_type is not None and rope_type in ALL_CUSTOM_ROPE_TYPES


# ---------------------------------------------------------------------------
# Cache baking
# ---------------------------------------------------------------------------

def bake_rope_caches(model: nn.Module, max_seq_len: int) -> Dict[str, torch.Tensor]:
    """Pre-compute RoPE cos/sin caches for every layer up to ``max_seq_len``.

    The model is expected to be an instance of ``LlamaForCausalLM`` (or any
    module that contains ``model.layers[*].self_attn.rotary_emb``).

    Args:
        model: The loaded LLaMA model with custom RoPE embeddings.
        max_seq_len: Maximum sequence length to bake caches for.

    Returns:
        A flat dictionary mapping tensor names to ``torch.Tensor`` values:
            ``rope_cos_cache.layer_{i}`` and ``rope_sin_cache.layer_{i}``.
        Each tensor has shape ``(max_seq_len, head_dim)`` and dtype float32.
    """
    caches: Dict[str, torch.Tensor] = {}

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError(
            "Model does not have the expected 'model.layers' structure. "
            "Ensure you are passing a LlamaForCausalLM instance."
        )

    layers = model.model.layers
    for layer_idx, layer in enumerate(layers):
        rotary_emb = layer.self_attn.rotary_emb

        # Create a dummy input tensor so the RoPE module can infer device/dtype.
        # Shape: (batch=1, num_heads=1, seq_len=max_seq_len, head_dim)
        head_dim = rotary_emb.dim if hasattr(rotary_emb, "dim") else rotary_emb.inv_freq.shape[0] * 2
        dummy = torch.zeros(1, 1, max_seq_len, head_dim, device=next(rotary_emb.parameters(), torch.tensor(0)).device)

        with torch.no_grad():
            cos, sin = rotary_emb(dummy, seq_len=max_seq_len)

        # Ensure consistent dtype / shape
        cos = cos[:max_seq_len].to(torch.float32).contiguous()
        sin = sin[:max_seq_len].to(torch.float32).contiguous()

        caches[f"rope_cos_cache_layer_{layer_idx}"] = cos
        caches[f"rope_sin_cache_layer_{layer_idx}"] = sin

    return caches


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_rope_metadata(config: LlamaConfig, model: nn.Module) -> Dict[str, Any]:
    """Extract all information needed to reconstruct the custom RoPE setup.

    The returned dictionary is JSON-serialisable and can be stored as GGUF
    metadata (string value) or expanded into individual ``custom_rope.*`` keys.

    Args:
        config: The model configuration (must be ``LlamaConfig``).
        model: The loaded model (used to grab per-layer RoPE attributes).

    Returns:
        A nested dictionary with keys such as:
            - ``type``, ``dynamic``, ``factor``
            - ``original_max_position_embeddings``, ``max_position_embeddings``
            - ``rope_theta``, ``num_hidden_layers``, ``head_dim``
            - ``alpha``, ``beta``, ``gamma`` (when present)
            - ``layers`` (list of per-layer dicts with ``i_star``, ``block_sizes``, …)
    """
    rope_scaling = getattr(config, "rope_scaling", None) or {}
    rope_type = rope_scaling.get("type", "none")

    meta: Dict[str, Any] = {
        "type": rope_type,
        "dynamic": bool(rope_scaling.get("dynamic", False)),
        "factor": float(rope_scaling.get("factor", 1.0)) if rope_scaling.get("factor") is not None else None,
        "original_max_position_embeddings": int(getattr(config, "original_max_position_embeddings", config.max_position_embeddings)),
        "max_position_embeddings": int(config.max_position_embeddings),
        "rope_theta": float(getattr(config, "rope_theta", 10000.0)),
        "num_hidden_layers": int(config.num_hidden_layers),
        "head_dim": _infer_head_dim(config, model),
    }

    # Global optional parameters
    for key in ("alpha", "beta", "gamma", "attn_scale_coef"):
        if key in rope_scaling:
            meta[key] = float(rope_scaling[key])

    # Per-layer attributes (critical for layer-aware / block-based methods)
    layers_meta: List[Dict[str, Any]] = []
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer_idx, layer in enumerate(model.model.layers):
            rotary_emb = layer.self_attn.rotary_emb
            layer_meta: Dict[str, Any] = {"layer_idx": layer_idx}

            # i_star
            if hasattr(rotary_emb, "i_star"):
                layer_meta["i_star"] = int(rotary_emb.i_star)

            # block_sizes (static mode only; dynamic mode will recompute)
            if hasattr(rotary_emb, "block_sizes"):
                bs = rotary_emb.block_sizes
                if isinstance(bs, torch.Tensor):
                    layer_meta["block_sizes"] = bs.cpu().tolist()

            # inv_freq (useful for debugging / verification)
            if hasattr(rotary_emb, "inv_freq"):
                inv = rotary_emb.inv_freq
                if isinstance(inv, torch.Tensor):
                    layer_meta["inv_freq"] = inv.cpu().tolist()

            # multi-scale buffers (my-rope2)
            if hasattr(rotary_emb, "scale_buffers"):
                scales = []
                for buf in rotary_emb.scale_buffers:
                    scales.append({
                        "window": int(buf.get("window", 0)),
                        "dim_start": int(buf.get("dim_start", 0)),
                        "dim_end": int(buf.get("dim_end", 0)),
                    })
                layer_meta["scales"] = scales

            # NTK params (my-rope2)
            if hasattr(rotary_emb, "_ntk_params"):
                layer_meta["ntk_params"] = [list(p) for p in rotary_emb._ntk_params]

            layers_meta.append(layer_meta)

    meta["layers"] = layers_meta
    return meta


def _infer_head_dim(config: LlamaConfig, model: nn.Module) -> int:
    """Infer head dimension from config or model structure."""
    if hasattr(config, "hidden_size") and hasattr(config, "num_attention_heads"):
        return config.hidden_size // config.num_attention_heads
    # Fallback: inspect first attention layer
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        first_attn = model.model.layers[0].self_attn
        if hasattr(first_attn, "head_dim"):
            return int(first_attn.head_dim)
    raise ValueError("Cannot infer head_dim from config or model.")


# ---------------------------------------------------------------------------
# GGUF I/O helpers
# ---------------------------------------------------------------------------

def write_custom_rope_metadata(gguf_writer, meta: Dict[str, Any]) -> None:
    """Write custom RoPE metadata into an open ``gguf.GGUFWriter``.

    All values are stored under the ``custom_rope.`` namespace.
    Complex nested structures are JSON-serialised.

    Args:
        gguf_writer: An instance of ``gguf.GGUFWriter`` (already initialised).
        meta: The dictionary returned by :func:`extract_rope_metadata`.
    """
    import gguf

    def _set(key: str, value: Any) -> None:
        full_key = f"custom_rope.{key}"
        if isinstance(value, bool):
            gguf_writer.add_bool(full_key, value)
        elif isinstance(value, int):
            gguf_writer.add_int32(full_key, value)
        elif isinstance(value, float):
            gguf_writer.add_float32(full_key, value)
        elif isinstance(value, str):
            gguf_writer.add_string(full_key, value)
        elif isinstance(value, (list, dict)):
            gguf_writer.add_string(full_key, json.dumps(value))
        else:
            gguf_writer.add_string(full_key, json.dumps(value))

    for key, value in meta.items():
        if key == "layers":
            # Store per-layer metadata as a JSON string under custom_rope.layers
            _set("layers", value)
        else:
            _set(key, value)


def read_custom_rope_metadata(gguf_path: str) -> Dict[str, Any]:
    """Read custom RoPE metadata from a GGUF file.

    Args:
        gguf_path: Path to the GGUF file.

    Returns:
        A dictionary mirroring the structure written by
        :func:`write_custom_rope_metadata`.
    """
    import gguf

    reader = gguf.GGUFReader(gguf_path)
    meta: Dict[str, Any] = {}

    for field in reader.fields.values():
        name: str = field.name
        if not name.startswith("custom_rope."):
            continue
        key = name[len("custom_rope."):]

        # Decode based on the field's declared GGUF value type.
        # field.parts is a list where the last element holds the raw value bytes.
        if not field.types:
            meta[key] = None
            continue

        val_type = field.types[0]
        raw_part = field.parts[-1]

        if val_type == gguf.GGUFValueType.STRING:
            raw = bytes(raw_part).decode("utf-8")
        elif val_type == gguf.GGUFValueType.BOOL:
            raw = bool(raw_part)
        elif val_type == gguf.GGUFValueType.INT32:
            raw = int(raw_part.item())
        elif val_type == gguf.GGUFValueType.FLOAT32:
            raw = float(raw_part.item())
        else:
            # Fallback for other numeric types
            if hasattr(raw_part, 'item'):
                raw = raw_part.item()
            else:
                raw = raw_part

        # Attempt JSON decode for complex structures (lists/dicts stored as strings)
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                pass
        meta[key] = raw

    return meta


# ---------------------------------------------------------------------------
# High-level helpers used by export_to_ollama.py
# ---------------------------------------------------------------------------

def prepare_export_state(model: nn.Module, config: LlamaConfig, max_seq_len: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Prepare everything needed for a custom-RoPE export.

    Returns:
        ``(extra_state_dict, rope_metadata)`` where:
        - ``extra_state_dict`` contains baked cos/sin caches (static mode) or
          is empty (dynamic mode).
        - ``rope_metadata`` is the serialisable metadata dict.
    """
    rope_scaling = getattr(config, "rope_scaling", None) or {}
    rope_type = rope_scaling.get("type", "none")
    is_dynamic = bool(rope_scaling.get("dynamic", False))

    meta = extract_rope_metadata(config, model)

    extra_state: Dict[str, torch.Tensor] = {}
    if _is_custom(rope_type) and not is_dynamic:
        print(f"[INFO] Baking RoPE caches for type='{rope_type}', max_seq_len={max_seq_len}")
        extra_state = bake_rope_caches(model, max_seq_len)
        print(f"[INFO] Baked {len(extra_state)} cache tensors.")
    elif _is_custom(rope_type) and is_dynamic:
        print(f"[INFO] Dynamic mode detected for type='{rope_type}'; skipping cache bake.")

    return extra_state, meta


def add_baked_caches_to_state_dict(state_dict: Dict[str, torch.Tensor], caches: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Merge baked RoPE caches into a model state_dict.

    Args:
        state_dict: Original model state dict.
        caches: Baked cache tensors from :func:`bake_rope_caches`.

    Returns:
        A new state dict containing both original weights and cache tensors.
    """
    merged = dict(state_dict)
    merged.update(caches)
    return merged
