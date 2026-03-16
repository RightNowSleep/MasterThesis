from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
import torch


def load_model_with_rope(config_path, rope_config_name):
    """
    Load a Llama model with specific RoPE extension based on configuration.
    """
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Find the corresponding RoPE configuration
    selected_rope = None
    for cfg in config["rope_configs"]:
        if cfg["name"] == rope_config_name:
            selected_rope = cfg["rope_scaling"]
            break

    if selected_rope is None:
        raise ValueError(f"RoPE config '{rope_config_name}' not found in config file.")

    model_name = config["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model configuration and apply RoPE settings
    model_config = LlamaConfig.from_pretrained(model_name)
    if selected_rope:
        model_config.rope_scaling = selected_rope
    else:
        model_config.rope_scaling = None

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=model_config,
        device_map="auto",  # Automatically use GPU
        torch_dtype=torch.float16,  # Save GPU memory
        output_attentions=True,  # Key: enable attention weight output
    )

    return model, tokenizer
