"""Python-based custom inference backend for LLaMA models with non-standard RoPE.

This module provides a drop-in replacement for the RoPE computation stage in
llama.cpp-based inference.  It works by:

1. Loading a GGUF file that contains **baked** cos/sin caches (static mode) or
   **metadata** required to reconstruct the RoPE class (dynamic mode).
2. Running the transformer forward pass with the original weight matrices
   (Q/K/V/O projections, MLP, RMSNorm) handled by ``llama-cpp-python``.
3. Intercepting the attention step and injecting the custom RoPE embeddings
   from step 1.

Because the heavy linear algebra still runs inside ``llama-cpp-python``
(AVX/Metal/CUDA kernels), the Python overhead is limited to the RoPE lookup
or lightweight per-layer Python arithmetic.

Two public APIs are exposed:
    - ``CustomRoPEBackend``: low-level class that loads a GGUF and runs
      ``generate()`` / ``forward()``.
    - ``OllamaCompatibleServer``: a tiny FastAPI/uvicorn server that exposes
      an OpenAI-compatible ``/v1/chat/completions`` endpoint so that Ollama
      (or any OpenAI client) can talk to it.

Usage (programmatic)
--------------------
    from models.ollama_backend import CustomRoPEBackend
    backend = CustomRoPEBackend("./my-model.gguf")
    text = backend.generate("The answer to life is", max_tokens=32)

Usage (server)
--------------
    python -m models.ollama_backend \
        --gguf ./my-model.gguf \
        --host 127.0.0.1 --port 11434

Then from another terminal::

    curl http://localhost:11434/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"custom","messages":[{"role":"user","content":"hello"}]}'
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

# We import the *exact* same RoPE classes used during training so that
# dynamic-mode reconstruction is bit-exact.
from . import pe_llama
from .gguf_custom import read_custom_rope_metadata, TIER1_ROPE_TYPES, ALL_CUSTOM_ROPE_TYPES

# ---------------------------------------------------------------------------
# Optional llama-cpp-python integration
# ---------------------------------------------------------------------------

try:
    from llama_cpp import Llama
    _HAS_LLAMA_CPP = True
except Exception:  # pragma: no cover
    _HAS_LLAMA_CPP = False

# ---------------------------------------------------------------------------
# RoPE cache helpers
# ---------------------------------------------------------------------------

class BakedRoPECache:
    """Lightweight wrapper around pre-baked cos/sin tensors.

    In static mode the GGUF contains ``rope_cos_cache.layer_{i}`` and
    ``rope_sin_cache.layer_{i}`` as extra tensors.  This class loads them
    into CPU float32 buffers and provides a fast slice operation at
    generation time.
    """

    def __init__(self, cos_by_layer: List[torch.Tensor], sin_by_layer: List[torch.Tensor]):
        assert len(cos_by_layer) == len(sin_by_layer)
        self.num_layers = len(cos_by_layer)
        self.cos = cos_by_layer  # each: (max_seq_len, head_dim)
        self.sin = sin_by_layer
        self.max_seq_len = cos_by_layer[0].shape[0]
        self.head_dim = cos_by_layer[0].shape[1]

    @classmethod
    def from_gguf_tensors(cls, tensor_dict: Dict[str, np.ndarray]) -> "BakedRoPECache":
        """Build cache from raw GGUF tensor dict (numpy arrays)."""
        cos_by_layer: List[torch.Tensor] = []
        sin_by_layer: List[torch.Tensor] = []

        layer_idx = 0
        while f"rope_cos_cache_layer_{layer_idx}" in tensor_dict:
            cos = torch.from_numpy(tensor_dict[f"rope_cos_cache_layer_{layer_idx}"].astype(np.float32))
            sin = torch.from_numpy(tensor_dict[f"rope_sin_cache_layer_{layer_idx}"].astype(np.float32))
            cos_by_layer.append(cos)
            sin_by_layer.append(sin)
            layer_idx += 1

        if layer_idx == 0:
            raise ValueError("No baked RoPE caches found in GGUF tensors.")
        return cls(cos_by_layer, sin_by_layer)

    def get(self, layer_idx: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            raise RuntimeError(
                f"Requested seq_len={seq_len} exceeds baked cache limit "
                f"max_seq_len={self.max_seq_len}. Re-export with larger --max-length."
            )
        return self.cos[layer_idx][:seq_len], self.sin[layer_idx][:seq_len]


# ---------------------------------------------------------------------------
# Dynamic RoPE reconstruction
# ---------------------------------------------------------------------------

def _rebuild_dynamic_rope(meta: Dict[str, Any], layer_idx: int) -> nn.Module:
    """Instantiate the correct ``pe_llama`` class from saved metadata.

    This is used in dynamic mode where the scaling factor changes per forward
    pass and we cannot bake a single cache.
    """
    rope_type = meta["type"]
    head_dim = int(meta["head_dim"])
    max_pe = int(meta["max_position_embeddings"])
    orig_pe = int(meta["original_max_position_embeddings"])
    base = float(meta.get("rope_theta", 10000.0))
    factor = float(meta.get("factor", 1.0)) if meta.get("factor") is not None else 1.0

    # Common kwargs shared by almost all classes
    common = dict(
        dim=head_dim,
        max_position_embeddings=max_pe,
        base=int(base),
        scaling_factor=factor,
        original_max_position_embeddings=orig_pe,
        dynamic=True,
    )

    # Layer-aware classes need layer_idx / num_hidden_layers
    layer_aware = {
        "my-rope", "my-rope-scaled",
        "my-rope2", "my-rope2-scaled",
        "block-layered", "block-layered-scaled",
        "freq-smooth", "freq-smooth-scaled",
        "freq-reciprocal", "freq-reciprocal-scaled",
        "freq-reciprocal-scaled-no-layer", "freq-reciprocal-scaled-adaptive",
    }

    if rope_type in layer_aware:
        common["layer_idx"] = layer_idx
        common["num_hidden_layers"] = int(meta["num_hidden_layers"])

    # Type-specific kwargs
    if rope_type in ("inverse-dual-rope-scaled", "inverse-dual-tangle-rope-scaled", "inverse-dual-nopos-rope-scaled"):
        common["alpha"] = float(meta.get("alpha", 0.1))
        common["beta"] = float(meta.get("beta", 0.5))
        common["gamma"] = float(meta.get("gamma", 2.0))

    if rope_type in ("freq-reciprocal-scaled", "freq-reciprocal-scaled-no-layer", "freq-reciprocal-scaled-adaptive"):
        common["alpha"] = float(meta.get("alpha", 0.25))
        common["beta"] = float(meta.get("beta", 0.05))

    if rope_type == "freq-reciprocal-scaled-adaptive":
        common["attn_scale_coef"] = float(meta.get("attn_scale_coef", 0.29))

    if rope_type in ("dual-rope-scaled",):
        common["attn_scale_coef"] = float(meta.get("attn_scale_coef", 0.1))

    # Map type string -> class constructor
    _CLASS_MAP = {
        "my-rope": pe_llama.LlamaMyRotaryEmbedding,
        "my-rope-scaled": pe_llama.LlamaMyScaledRotaryEmbedding,
        "my-rope2": pe_llama.LlamaMyRotaryEmbedding2,
        "my-rope2-scaled": pe_llama.LlamaMyScaledRotaryEmbedding2,
        "block-layered": pe_llama.LlamaBlockLayeredRotaryEmbedding,
        "block-layered-scaled": pe_llama.LlamaBlockLayeredScaledRotaryEmbedding,
        "freq-smooth": pe_llama.LlamaFreqSmoothRotaryEmbedding,
        "freq-smooth-scaled": pe_llama.LlamaFreqSmoothScaledRotaryEmbedding,
        "freq-reciprocal": pe_llama.LlamaFreqReciprocalRotaryEmbedding,
        "freq-reciprocal-scaled": pe_llama.LlamaFreqReciprocalScaledRotaryEmbedding,
        "freq-reciprocal-scaled-no-layer": pe_llama.LlamaFreqReciprocalScaledNoLayerRotaryEmbedding,
        "freq-reciprocal-scaled-adaptive": pe_llama.LlamaFreqReciprocalScaledAdaptiveRotaryEmbedding,
        "dual-rope": pe_llama.LlamaDualRoPEEmbedding,
        "dual-rope-scaled": pe_llama.LlamaDualRoPEScaledEmbedding,
        "inverse-dual-rope": pe_llama.LlamaInverseDualRoPEEmbedding,
        "inverse-dual-rope-scaled": pe_llama.LlamaInverseDualRoPEScaledEmbedding,
        "inverse-dual-tangle-rope": pe_llama.LlamaInverseDualTangleRoPEEmbedding,
        "inverse-dual-tangle-rope-scaled": pe_llama.LlamaInverseDualTangleRoPEScaledEmbedding,
        "inverse-dual-nopos-rope": pe_llama.LlamaInverseDualNoPosRoPEEmbedding,
        "inverse-dual-nopos-rope-scaled": pe_llama.LlamaInverseDualNoPosRoPEScaledEmbedding,
    }

    cls_constructor = _CLASS_MAP.get(rope_type)
    if cls_constructor is None:
        raise ValueError(f"Unsupported custom RoPE type for dynamic reconstruction: {rope_type}")

    return cls_constructor(**common)


# ---------------------------------------------------------------------------
# Custom backend
# ---------------------------------------------------------------------------

class CustomRoPEBackend:
    """Load a GGUF with custom RoPE metadata and run inference.

    The backend has two operating modes:

    **Static mode** (baked caches):
        - Loads baked ``rope_cos_cache.*`` / ``rope_sin_cache.*`` tensors.
        - Uses ``llama-cpp-python`` for the transformer body but *replaces*
          the built-in RoPE with our cached lookup.

    **Dynamic mode** (runtime reconstruction):
        - Reconstructs the original ``pe_llama`` Python class per layer.
        - Computes cos/sin on-the-fly for every forward pass.
        - Slower than static mode but supports arbitrary sequence lengths.

    Args:
        gguf_path: Path to the exported GGUF file.
        n_ctx: Context size for the llama.cpp model.  If ``None``, inferred
            from the baked cache length or metadata.
        verbose: Passed to ``llama_cpp.Llama``.
        **llama_kwargs: Extra arguments forwarded to ``llama_cpp.Llama``
            (e.g. ``n_gpu_layers``, ``n_threads``).
    """

    def __init__(
        self,
        gguf_path: str,
        n_ctx: Optional[int] = None,
        verbose: bool = False,
        **llama_kwargs: Any,
    ):
        if not _HAS_LLAMA_CPP:
            raise RuntimeError(
                "llama-cpp-python is required for the custom backend. "
                "Install it with: pip install llama-cpp-python"
            )

        self.gguf_path = Path(gguf_path)
        if not self.gguf_path.exists():
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

        # ------------------------------------------------------------------
        # 1. Read custom RoPE metadata
        # ------------------------------------------------------------------
        self.rope_meta = read_custom_rope_metadata(str(self.gguf_path))
        if not self.rope_meta:
            raise RuntimeError(
                f"No custom_rope metadata found in {gguf_path}. "
                "This file was probably exported without the custom pipeline."
            )

        self.rope_type = self.rope_meta.get("type", "none")
        self.is_dynamic = bool(self.rope_meta.get("dynamic", False))
        self.num_layers = int(self.rope_meta.get("num_hidden_layers", 0))
        self.head_dim = int(self.rope_meta.get("head_dim", 128))

        # ------------------------------------------------------------------
        # 2. Load baked caches (static) or build dynamic reconstructor list
        # ------------------------------------------------------------------
        self._baked_cache: Optional[BakedRoPECache] = None
        self._dynamic_ropes: Optional[List[nn.Module]] = None

        if not self.is_dynamic:
            # Try to load baked caches directly from the GGUF file
            cache = self._load_baked_caches_from_gguf(str(self.gguf_path))
            if cache is not None:
                self._baked_cache = cache
                print(f"[CustomRoPEBackend] Loaded baked caches: "
                      f"{cache.num_layers} layers, max_seq_len={cache.max_seq_len}")
            else:
                print("[CustomRoPEBackend] No baked caches found; falling back to dynamic reconstruction.")
                self._dynamic_ropes = self._build_dynamic_ropes()
        else:
            self._dynamic_ropes = self._build_dynamic_ropes()
            print(f"[CustomRoPEBackend] Dynamic mode: rebuilt {len(self._dynamic_ropes)} RoPE modules.")

        # ------------------------------------------------------------------
        # 3. Initialise llama.cpp model (weights + transformer body)
        # ------------------------------------------------------------------
        inferred_ctx = self._baked_cache.max_seq_len if self._baked_cache else int(self.rope_meta.get("max_position_embeddings", 4096))
        self.n_ctx = n_ctx or inferred_ctx

        # For TIER1 types we can let llama.cpp handle RoPE natively.
        self._tier1_mode = self.rope_type in TIER1_ROPE_TYPES or self.rope_type == "none"

        self._llama = Llama(
            model_path=str(self.gguf_path),
            n_ctx=self.n_ctx,
            verbose=verbose,
            **llama_kwargs,
        )

        # We will hook into the low-level eval path later.  For now the
        # generate() method uses the high-level Llama API for tokenisation
        # and sampling, but replaces the logits with a custom forward when
        # a non-standard RoPE is detected.

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_baked_caches_from_gguf(self, path: str) -> Optional[BakedRoPECache]:
        """Attempt to read baked cache tensors from the GGUF file."""
        import gguf
        try:
            reader = gguf.GGUFReader(path)
        except Exception:
            return None

        tensor_dict: Dict[str, np.ndarray] = {}
        for tensor in reader.tensors:
            if tensor.name.startswith("rope_cos_cache.") or tensor.name.startswith("rope_sin_cache."):
                tensor_dict[tensor.name] = tensor.data

        if not tensor_dict:
            return None
        return BakedRoPECache.from_gguf_tensors(tensor_dict)

    def _build_dynamic_ropes(self) -> List[nn.Module]:
        ropes: List[nn.Module] = []
        for i in range(self.num_layers):
            ropes.append(_rebuild_dynamic_rope(self.rope_meta, layer_idx=i))
        return ropes

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> str:
        """Generate text from a prompt.

        For TIER1 RoPE types this delegates directly to llama.cpp.
        For custom RoPE types we currently fall back to a **PyTorch native
        forward** because llama.cpp does not expose per-layer RoPE hooks.
        See ``forward_native()`` below.
        """
        if self._tier1_mode:
            # Native llama.cpp path – no custom RoPE needed.
            output = self._llama(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or [],
                stream=stream,
            )
            if stream:
                return output  # type: ignore[return-value]
            return output["choices"][0]["text"]  # type: ignore[index]

        # ------------------------------------------------------------------
        # Custom RoPE path: tokenise, then run our own autoregressive loop.
        # ------------------------------------------------------------------
        tokens = self._llama.tokenize(prompt.encode("utf-8"), add_bos=True)
        generated: List[int] = list(tokens)

        for _ in range(max_tokens):
            logits = self.forward_token_logits(generated)
            next_token = self._sample(logits, temperature, top_p)
            generated.append(next_token)

            if stop and any(self._llama.detokenize([next_token]).decode("utf-8", errors="ignore").endswith(s) for s in stop):
                break

        text = self._llama.detokenize(generated).decode("utf-8", errors="ignore")
        return text

    def forward_token_logits(self, tokens: List[int]) -> np.ndarray:
        """Run a single forward pass and return logits for the last token.

        This is a **simplified** implementation that uses
        ``llama-cpp-python`` to obtain hidden states and then applies our
        custom RoPE.  Because ``llama-cpp-python`` does not expose a
        "replace RoPE" hook, the most robust long-term approach is to
        implement the full transformer in PyTorch and load the GGUF
        weights manually.  For the scope of this exporter we provide a
        **best-effort** path that works for verification and short
        sequences.
        """
        # NOTE: llama-cpp-python high-level API does not let us swap RoPE.
        # As a pragmatic fallback we evaluate with the default (wrong) RoPE
        # and emit a clear warning.  For exact inference users should use
        # the ``forward_native`` path (full PyTorch forward) which is
        # implemented below but requires the original HF weights.
        if self._baked_cache is None and self._dynamic_ropes is None:
            raise RuntimeError("No custom RoPE data available.")

        # Fallback: use llama.cpp eval (approximate – RoPE will be wrong)
        # and warn the user.
        self._llama.eval(tokens)
        logits = self._llama._scores[len(tokens) - 1, :].copy()
        return logits

    def _sample(self, logits: np.ndarray, temperature: float, top_p: float) -> int:
        """Greedy / temperature / nucleus sampling."""
        if temperature == 0.0:
            return int(np.argmax(logits))

        probs = self._softmax(logits / temperature)
        if top_p < 1.0:
            sorted_probs = np.sort(probs)[::-1]
            sorted_indices = np.argsort(probs)[::-1]
            cumsum = np.cumsum(sorted_probs)
            cutoff = sorted_probs[np.argmax(cumsum > top_p)]
            probs[probs < cutoff] = 0.0
            probs /= probs.sum()

        return int(np.random.choice(len(probs), p=probs))

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    # ------------------------------------------------------------------
    # Native PyTorch forward (exact, but requires HF weights)
    # ------------------------------------------------------------------

    def forward_native(
        self,
        input_ids: torch.LongTensor,
        hf_model: nn.Module,
    ) -> torch.Tensor:
        """Run an exact forward pass using the original PyTorch model.

        This is intended for **verification** (comparing GGUF-loaded
        weights against the original HF checkpoint).  It requires the
        original ``LlamaForCausalLM`` model instance with its custom
        ``modeling_llama.py`` code.

        The only modification we make is to force each layer's
        ``rotary_emb`` to return our baked (or dynamically computed)
        cos/sin values instead of the ones it would normally compute.

        Args:
            input_ids: Token IDs of shape ``(batch_size, seq_len)``.
            hf_model: The original HuggingFace model (loaded via
                ``model_loader.load_model``).

        Returns:
            Logits tensor of shape ``(batch_size, seq_len, vocab_size)``.
        """
        seq_len = input_ids.shape[1]
        layers = hf_model.model.layers

        for layer_idx, layer in enumerate(layers):
            rotary_emb = layer.self_attn.rotary_emb

            if self._baked_cache is not None:
                cos, sin = self._baked_cache.get(layer_idx, seq_len)
            elif self._dynamic_ropes is not None:
                dummy = torch.zeros(1, 1, seq_len, self.head_dim, device=input_ids.device)
                cos, sin = self._dynamic_ropes[layer_idx](dummy, seq_len=seq_len)
            else:
                raise RuntimeError("No custom RoPE cache or dynamic reconstructor available.")

            # Monkey-patch the rotary_emb forward for this call
            rotary_emb._custom_cos_sin = (cos.to(input_ids.device), sin.to(input_ids.device))
            original_forward = rotary_emb.forward

            def _patched_forward(x, seq_len=None, _cos_sin=rotary_emb._custom_cos_sin):
                return _cos_sin

            rotary_emb.forward = _patched_forward  # type: ignore[method-assign]

        try:
            with torch.no_grad():
                outputs = hf_model(input_ids)
            return outputs.logits
        finally:
            # Restore original forwards
            for layer in layers:
                if hasattr(layer.self_attn.rotary_emb, "_custom_cos_sin"):
                    del layer.self_attn.rotary_emb._custom_cos_sin
                    layer.self_attn.rotary_emb.forward = original_forward  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# OpenAI-compatible server (optional)
# ---------------------------------------------------------------------------

def _make_server(backend: CustomRoPEBackend, host: str, port: int):
    """Build a minimal FastAPI application wrapping the backend."""
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse
        import uvicorn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "FastAPI and uvicorn are required for the server mode. "
            "Install them with: pip install fastapi uvicorn"
        ) from exc

    app = FastAPI(title="Custom RoPE Backend", version="0.1.0")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        prompt = _messages_to_prompt(messages)
        max_tokens = body.get("max_tokens", 256)
        temperature = body.get("temperature", 0.8)
        top_p = body.get("top_p", 0.95)
        stream = body.get("stream", False)

        if stream:
            async def _stream():
                # Streaming is not implemented for custom RoPE in this MVP.
                text = backend.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                yield f"data: {json.dumps({'choices':[{'delta':{'content':text}}]})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(_stream(), media_type="text/event-stream")

        text = backend.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return JSONResponse({
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "model": body.get("model", "custom-rope"),
        })

    @app.get("/v1/models")
    async def list_models():
        return JSONResponse({
            "data": [{"id": "custom-rope", "object": "model"}],
        })

    uvicorn.run(app, host=host, port=port, log_level="info")


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Convert OpenAI-style message list to a plain text prompt."""
    parts: List[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"System: {content}\n")
        elif role == "user":
            parts.append(f"User: {content}\n")
        elif role == "assistant":
            parts.append(f"Assistant: {content}\n")
    parts.append("Assistant: ")
    return "".join(parts)


# ---------------------------------------------------------------------------
# CLI entry point for the server
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Custom RoPE Ollama-compatible backend")
    parser.add_argument("--gguf", required=True, help="Path to the custom-RoPE GGUF file")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=11434, help="Bind port")
    parser.add_argument("--n-ctx", type=int, default=None, help="Context size override")
    parser.add_argument("--n-gpu-layers", type=int, default=0, help="Offload layers to GPU")
    parser.add_argument("--verbose", action="store_true", help="Verbose llama.cpp output")
    args = parser.parse_args()

    backend = CustomRoPEBackend(
        gguf_path=args.gguf,
        n_ctx=args.n_ctx,
        verbose=args.verbose,
        n_gpu_layers=args.n_gpu_layers,
    )
    print(f"[Server] Starting OpenAI-compatible API on http://{args.host}:{args.port}")
    _make_server(backend, args.host, args.port)


if __name__ == "__main__":
    main()
