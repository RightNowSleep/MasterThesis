from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
import torch


def load_model_with_rope(config_path, rope_config_name):
    """
    根据配置加载带有特定RoPE扩展的Llama模型。
    """
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 找到对应的rope配置
    selected_rope = None
    for cfg in config['rope_configs']:
        if cfg['name'] == rope_config_name:
            selected_rope = cfg['rope_scaling']
            break

    if selected_rope is None:
        raise ValueError(f"RoPE config '{rope_config_name}' not found in config file.")

    model_name = config['base_model']
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 加载模型配置并应用RoPE设置
    model_config = LlamaConfig.from_pretrained(model_name)
    if selected_rope:
        model_config.rope_scaling = selected_rope
    else:
        model_config.rope_scaling = None

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=model_config,
        device_map="auto",  # 自动使用GPU
        torch_dtype=torch.float16,  # 节省显存
        output_attentions=True  # 关键：启用注意力权重输出
    )

    return model, tokenizer