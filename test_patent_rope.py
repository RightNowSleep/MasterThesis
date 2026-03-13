"""
单元测试 + 使用示例
把 PatentRotaryEmbedding 注入 transformers 的任意 RoPE 模型
"""
import torch, math
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3ForCausalLM, Qwen3Config, Llama4ForCausalLM
from modeling_rope_patent import PatentRotaryEmbedding
from transformers.modeling_rope_utils import get_rope

# 1. 注册进工厂函数（一次性）
def _patent_rope_hook(dim, max_seq_len, base, rope_scaling, device):
    return PatentRotaryEmbedding(
        dim=dim,
        max_seq_len=max_seq_len,
        base=base,
        N=rope_scaling["N"],        # 模型层数
        L=rope_scaling["L"],        # 原始窗口
        alpha=rope_scaling["alpha"],
        device=device,
    )

# 偷懒写法：直接 monkey-patch
import transformers.modeling_rope_utils as ru
ru._rope_type_registry["patent"] = _patent_rope_hook

# 2. 加载模型（以 Qwen2.5-7B 为例）
model_id = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    rope_scaling={
        "rope_type": "patent",
        "N": 32,      # Qwen2.5 总层数
        "L": 4096,    # 原始窗口
        "alpha": 0.2,
    },
)

# 3. 推理 demo：32 k token
prompt = "你好 " * 16000
inputs = tok(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=50)
print(tok.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))

# 4. 数值回归测试（专利 2.2 手工算例）
def test_w_ext():
    pe = PatentRotaryEmbedding(dim=128, max_seq_len=512, N=4, L=256)
    assert pe.w_ext.shape == (4, 128)
    # 高频维度 r(d)<=1 强制为 0
    r = 256 / (2 * math.pi / (10000 ** (-torch.arange(0, 128, 2) / 128)))
    assert (pe.w_ext[:, r <= 1] == 0.0).all()
    print("✓ w_ext 高频归零测试通过")

def test_temperature_scale():
    pe = PatentRotaryEmbedding(dim=64, max_seq_len=8192, L=4096, alpha=0.2)
    scale = pe._get_attention_scale(8192, device="cpu")
    assert scale.shape == (8192, 1)
    # 位置 4096 之后单调增
    assert torch.all(scale[4096:] > scale[4095])
    print("✓ 温度缩放单调增测试通过")

if __name__ == "__main__":
    test_w_ext()
    test_temperature_scale()
    print("全部单元测试通过！")