"""
Unit tests + usage examples
Inject PatentRotaryEmbedding into any RoPE model in transformers
"""

import torch, math
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen3ForCausalLM,
    Qwen3Config,
    Llama4ForCausalLM,
)
from modeling_rope_patent import PatentRotaryEmbedding
from transformers.modeling_rope_utils import get_rope


# 1. Register into factory function (one-time)
def _patent_rope_hook(dim, max_seq_len, base, rope_scaling, device):
    return PatentRotaryEmbedding(
        dim=dim,
        max_seq_len=max_seq_len,
        base=base,
        N=rope_scaling["N"],  # Number of model layers
        L=rope_scaling["L"],  # Original window size
        alpha=rope_scaling["alpha"],
        device=device,
    )


# Shortcut: direct monkey-patch
import transformers.modeling_rope_utils as ru

ru._rope_type_registry["patent"] = _patent_rope_hook

# 2. Load model (using Qwen2.5-7B as example)
model_id = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    rope_scaling={
        "rope_type": "patent",
        "N": 32,  # Qwen2.5 total layers
        "L": 4096,  # Original window size
        "alpha": 0.2,
    },
)

# 3. Inference demo: 32k tokens
prompt = "Hello " * 16000
inputs = tok(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=50)
print(tok.decode(out[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True))


# 4. Numerical regression test (patent 2.2 manual calculation example)
def test_w_ext():
    pe = PatentRotaryEmbedding(dim=128, max_seq_len=512, N=4, L=256)
    assert pe.w_ext.shape == (4, 128)
    # High-frequency dimensions r(d)<=1 forced to 0
    r = 256 / (2 * math.pi / (10000 ** (-torch.arange(0, 128, 2) / 128)))
    assert (pe.w_ext[:, r <= 1] == 0.0).all()
    print("✓ w_ext high-frequency zeroing test passed")


def test_temperature_scale():
    pe = PatentRotaryEmbedding(dim=64, max_seq_len=8192, L=4096, alpha=0.2)
    scale = pe._get_attention_scale(8192, device="cpu")
    assert scale.shape == (8192, 1)
    # Temperature scale monotonically increasing after position 4096
    assert torch.all(scale[4096:] > scale[4095])
    print("✓ Temperature scaling monotonic increase test passed")


if __name__ == "__main__":
    test_w_ext()
    test_temperature_scale()
    print("All unit tests passed!")
