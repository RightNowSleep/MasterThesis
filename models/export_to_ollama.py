"""Export a LLaMA model with custom RoPE to an Ollama-compatible GGUF format.

This script implements the full export pipeline described in the project plan:

    1. Load the model + tokenizer via ``model_loader`` (supports all 3 modes).
    2. Determine RoPE tier (TIER1 native vs TIER2/3 custom).
    3. For custom static RoPE: bake cos/sin caches and merge into state dict.
    4. Save as HuggingFace checkpoint (config + weights + optional caches).
    5. Convert to GGUF using the standard ``convert_hf_to_gguf.py`` script.
    6. Inject ``custom_rope.*`` metadata into the GGUF file.
    7. Generate an Ollama ``Modelfile``.
    8. Optionally run ``ollama create`` to import the model.

CLI Example
-----------
    python models/export_to_ollama.py \\
        --model-name meta-llama/Llama-2-7b-hf \\
        --adapter-path ./checkpoints/my-adapter \\
        --output-dir ./ollama-export \\
        --ollama-model-name my-custom-rope \\
        --quant Q4_K_M \\
        --max-length 8192

Requirements
------------
    - ``gguf`` package (pip install gguf)
    - ``llama.cpp`` repository with ``convert_hf_to_gguf.py`` on PATH or
      specified via ``--convert-script``
    - ``ollama`` CLI installed (optional, for final import step)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from .model_loader import (
    add_args_model,
    load_model,
    load_tokenizer,
    _build_rope_scaling,
)
from .gguf_custom import (
    TIER1_ROPE_TYPES,
    ALL_CUSTOM_ROPE_TYPES,
    prepare_export_state,
    add_baked_caches_to_state_dict,
    write_custom_rope_metadata,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_QUANTS = [
    "Q4_0",
    "Q4_K_M",
    "Q5_0",
    "Q5_K_M",
    "Q6_K",
    "Q8_0",
    "F16",
    "F32",
]

DEFAULT_QUANT = "Q4_K_M"

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a LLaMA model with custom RoPE to Ollama-compatible GGUF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Re-use all model-loading arguments from model_loader
    add_args_model(parser)

    # Export-specific arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where the exported artefacts (HF checkpoint, GGUF, Modelfile) will be written.",
    )
    parser.add_argument(
        "--ollama-model-name",
        type=str,
        default=None,
        help="Name for the Ollama model.  Defaults to the basename of --output-dir.",
    )
    parser.add_argument(
        "--quant",
        type=str,
        default=DEFAULT_QUANT,
        choices=VALID_QUANTS,
        help=f"GGUF quantization type.  Default: {DEFAULT_QUANT}",
    )
    parser.add_argument(
        "--convert-script",
        type=str,
        default=None,
        help=(
            "Path to the llama.cpp ``convert_hf_to_gguf.py`` script.  "
            "If not provided, the script is searched on PATH."
        ),
    )
    parser.add_argument(
        "--skip-ollama-create",
        action="store_true",
        help="Skip the final 'ollama create' step (useful when Ollama is not installed locally).",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt written into the generated Modelfile.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Default sampling temperature for the Modelfile.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Default nucleus-sampling top_p for the Modelfile.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help=(
            "Jinja2 chat template string for the Modelfile.  "
            "If omitted, a generic template is generated."
        ),
    )
    parser.add_argument(
        "--use-safetensors",
        action="store_true",
        help="Save HF checkpoint weights in Safetensors format instead of PyTorch .bin files.",
    )

    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_convert_script(user_path: Optional[str]) -> str:
    """Locate ``convert_hf_to_gguf.py`` or raise a clear error."""
    if user_path is not None:
        p = Path(user_path)
        if p.exists():
            return str(p.resolve())
        raise FileNotFoundError(f"--convert-script not found: {user_path}")

    # Search on PATH
    for candidate in ("convert_hf_to_gguf.py", "convert_hf_to_gguf"):
        found = shutil.which(candidate)
        if found:
            return found

    # Common locations inside a llama.cpp checkout
    home = Path.home()
    common_paths = [
        home / "llama.cpp" / "convert_hf_to_gguf.py",
        home / "llama.cpp" / "gguf-py" / "scripts" / "convert_hf_to_gguf.py",
        Path("/usr/local/src/llama.cpp/convert_hf_to_gguf.py"),
        Path("/opt/llama.cpp/convert_hf_to_gguf.py"),
    ]
    for p in common_paths:
        if p.exists():
            return str(p.resolve())

    raise RuntimeError(
        "Could not find convert_hf_to_gguf.py. "
        "Please clone llama.cpp and point --convert-script to it, "
        "or ensure the script is on your PATH."
    )


def _save_hf_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: Path,
    use_safetensors: bool = False,
) -> None:
    """Save model + tokenizer as a HuggingFace-format checkpoint.

    Args:
        model: The loaded (and optionally merged) model.
        tokenizer: The corresponding tokenizer.
        output_dir: Destination directory.
        use_safetensors: If True, use ``safetensors`` instead of ``pytorch_model.bin``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    tokenizer.save_pretrained(output_dir)

    # Config
    model.config.save_pretrained(output_dir)

    # Weights
    if use_safetensors:
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise RuntimeError(
                "safetensors is required for --use-safetensors. "
                "Install it with: pip install safetensors"
            ) from exc
        save_file(model.state_dict(), output_dir / "model.safetensors")
    else:
        torch.save(model.state_dict(), output_dir / "pytorch_model.bin")

    print(f"[HF] Checkpoint saved to {output_dir}")


def _run_gguf_conversion(
    hf_dir: Path,
    gguf_out: Path,
    quant: str,
    convert_script: str,
) -> None:
    """Invoke ``convert_hf_to_gguf.py`` to produce a quantized GGUF."""
    cmd: List[str] = [
        sys.executable,
        convert_script,
        str(hf_dir),
        "--outfile",
        str(gguf_out),
        "--outtype",
        quant,
    ]
    print(f"[GGUF] Running conversion: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"GGUF conversion failed with exit code {result.returncode}")
    print(f"[GGUF] Conversion complete: {gguf_out}")


def _inject_custom_metadata(gguf_path: Path, meta: Dict[str, Any]) -> None:
    """Append custom_rope metadata to an existing GGUF file.

    Because ``convert_hf_to_gguf.py`` does not know about our custom keys,
    we open the file in-place with ``gguf.GGUFWriter`` and add them after
    the conversion step.
    """
    import gguf

    # GGUFWriter currently only supports creating new files, so we use a
    # read-modify-write approach via GGUFReader + a temporary writer.
    reader = gguf.GGUFReader(str(gguf_path))

    # Determine architecture / alignment from existing file
    arch = "llama"
    try:
        arch_field = reader.get_field("general.architecture")
        if arch_field is not None:
            arch = str(arch_field.parts[0])
    except Exception:
        pass

    tmp_path = gguf_path.with_suffix(".gguf.tmp")
    writer = gguf.GGUFWriter(str(tmp_path), arch)

    # Copy all existing metadata (skip GGUF internal keys that the writer sets automatically)
    _gguf_internal_keys = {
        "GGUF.version",
        "GGUF.tensor_count",
        "GGUF.kv_count",
        "general.architecture",
        "general.name",
    }
    for field in reader.fields.values():
        name = field.name
        if name in _gguf_internal_keys:
            continue
        val = field.parts
        # field.parts is a numpy array (or memmap) for scalar/string data, but may be a list for arrays
        if hasattr(val, 'ndim'):
            if val.ndim == 0 or val.size == 1:
                raw = val.item() if val.size == 1 else val.tolist()
            else:
                raw = val.tolist()
        else:
            raw = val
        # Re-add with correct type inference
        if isinstance(raw, bool):
            writer.add_bool(name, raw)
        elif isinstance(raw, int):
            writer.add_int32(name, raw)
        elif isinstance(raw, float):
            writer.add_float32(name, raw)
        elif isinstance(raw, str):
            writer.add_string(name, raw)
        else:
            # Fallback: convert numpy memmap / array to plain Python list first
            try:
                import numpy as np
                if isinstance(raw, (np.ndarray, np.memmap)):
                    raw = raw.tolist()
            except Exception:
                pass
            # Final safety: if still not JSON-serialisable, force string repr
            try:
                json.dumps(raw)
            except TypeError:
                raw = str(raw)
            writer.add_string(name, raw)

    # Copy all existing tensors
    for tensor in reader.tensors:
        writer.add_tensor(tensor.name, tensor.data, raw_shape=tensor.shape)

    # Inject custom RoPE metadata
    write_custom_rope_metadata(writer, meta)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # Atomic replace
    tmp_path.replace(gguf_path)
    print(f"[GGUF] Injected custom_rope metadata into {gguf_path}")


def _generate_modelfile(
    gguf_path: Path,
    output_path: Path,
    model_name: str,
    system_prompt: str,
    temperature: float,
    top_p: float,
    template: Optional[str],
    rope_meta: Dict[str, Any],
) -> None:
    """Generate a standard Ollama Modelfile."""
    if template is None:
        # Generic chat template
        template = (
            "{{ if .System }}<|system|>\n{{ .System }}\n{{ end }}"
            "{{ if .Prompt }}<|user|>\n{{ .Prompt }}\n{{ end }}"
            "<|assistant|>\n{{ .Response }}"
        )

    lines = [
        f"FROM {gguf_path.name}",
        "",
        f"SYSTEM \"{system_prompt}\"",
        "",
        f"PARAMETER temperature {temperature}",
        f"PARAMETER top_p {top_p}",
        "",
        f"TEMPLATE \"{template}\"",
        "",
        "# Custom RoPE metadata (for reference)",
        f"# rope_type: {rope_meta.get('type', 'none')}",
        f"# dynamic: {rope_meta.get('dynamic', False)}",
        f"# factor: {rope_meta.get('factor', 'N/A')}",
        f"# max_position_embeddings: {rope_meta.get('max_position_embeddings', 'N/A')}",
        f"# original_max_position_embeddings: {rope_meta.get('original_max_position_embeddings', 'N/A')}",
    ]

    if rope_meta.get("alpha") is not None:
        lines.append(f"# alpha: {rope_meta['alpha']}")
    if rope_meta.get("beta") is not None:
        lines.append(f"# beta: {rope_meta['beta']}")
    if rope_meta.get("gamma") is not None:
        lines.append(f"# gamma: {rope_meta['gamma']}")

    lines.append("")
    lines.append(
        "# NOTE: For TIER2/3 custom RoPE models, use the Python backend:"
    )
    lines.append(
        "#   python -m models.ollama_backend --gguf ./model.gguf"
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Modelfile] Generated {output_path}")


def _run_ollama_create(modelfile: Path, model_name: str) -> None:
    """Execute ``ollama create`` to import the model."""
    if shutil.which("ollama") is None:
        print("[WARNING] 'ollama' CLI not found on PATH. Skipping ollama create.")
        return

    cmd = ["ollama", "create", model_name, "-f", str(modelfile)]
    print(f"[Ollama] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"ollama create failed with exit code {result.returncode}")
    print(f"[Ollama] Model '{model_name}' created successfully.")


# ---------------------------------------------------------------------------
# Main export pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 0. Validate arguments
    # ------------------------------------------------------------------
    if args.ollama_model_name is None:
        args.ollama_model_name = Path(args.output_dir).name

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load model + tokenizer
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1/6: Loading model and tokenizer")
    print("=" * 60)
    model, config = load_model(args)
    tokenizer = load_tokenizer(args)

    rope_scaling = getattr(config, "rope_scaling", None) or {}
    rope_type = rope_scaling.get("type", "none")
    is_dynamic = bool(rope_scaling.get("dynamic", False))

    print(f"[INFO] Detected RoPE type: {rope_type}")
    print(f"[INFO] Dynamic mode: {is_dynamic}")

    # ------------------------------------------------------------------
    # 2. Prepare export state (caches + metadata)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 2/6: Preparing export state")
    print("=" * 60)
    extra_state, rope_meta = prepare_export_state(model, config, args.max_length)

    # Merge baked caches into the model state dict if present
    if extra_state:
        # We need to attach the caches so that save_pretrained writes them.
        # The simplest way is to register them as buffers on the model.
        for name, tensor in extra_state.items():
            model.register_buffer(name, tensor, persistent=True)
        print(f"[INFO] Registered {len(extra_state)} baked cache buffers on model.")

    # ------------------------------------------------------------------
    # 3. Save HF checkpoint
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 3/6: Saving HuggingFace checkpoint")
    print("=" * 60)
    hf_dir = output_dir / "hf_checkpoint"
    _save_hf_checkpoint(model, tokenizer, hf_dir, use_safetensors=args.use_safetensors)

    # ------------------------------------------------------------------
    # 4. Convert to GGUF
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 4/6: Converting to GGUF")
    print("=" * 60)
    convert_script = _find_convert_script(args.convert_script)
    gguf_path = output_dir / f"{args.ollama_model_name}.{args.quant.lower()}.gguf"
    _run_gguf_conversion(hf_dir, gguf_path, args.quant, convert_script)

    # ------------------------------------------------------------------
    # 5. Inject custom metadata
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 5/6: Injecting custom RoPE metadata")
    print("=" * 60)
    if rope_type in ALL_CUSTOM_ROPE_TYPES:
        _inject_custom_metadata(gguf_path, rope_meta)
    else:
        print("[INFO] TIER1 RoPE type detected; no custom metadata injection needed.")

    # ------------------------------------------------------------------
    # 6. Generate Modelfile and optionally create Ollama model
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 6/6: Generating Modelfile")
    print("=" * 60)
    modelfile_path = output_dir / "Modelfile"
    _generate_modelfile(
        gguf_path=gguf_path,
        output_path=modelfile_path,
        model_name=args.ollama_model_name,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        template=args.template,
        rope_meta=rope_meta,
    )

    if not args.skip_ollama_create:
        _run_ollama_create(modelfile_path, args.ollama_model_name)
    else:
        print("[INFO] Skipped ollama create (--skip-ollama-create).")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Export complete!")
    print("=" * 60)
    print(f"  HF checkpoint : {hf_dir}")
    print(f"  GGUF          : {gguf_path}")
    print(f"  Modelfile     : {modelfile_path}")
    print(f"  RoPE type     : {rope_type}")
    print(f"  Quant         : {args.quant}")
    if rope_type in ALL_CUSTOM_ROPE_TYPES:
        print("")
        print("  NOTE: This model uses a custom RoPE type.")
        print("        To run inference with exact RoPE semantics, start the Python backend:")
        print(f"          python -m models.ollama_backend --gguf {gguf_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
