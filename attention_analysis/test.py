from src.model_loader import load_model_with_rope
from src.attention_extractor import extract_attention_from_model
from src.visualizer import plot_heatmaps_with_bertviz, plot_static_heatmap
import yaml
import torch

# 1. Load configuration
config_path = "../config/model_config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# 2. Prepare input data
with open("../data/your_dataset.txt", "r") as f:
    input_text = f.read().strip()

# 3. Iterate through all RoPE configurations
for rope_cfg in config["rope_configs"]:
    print(f"\n--- Analyzing {rope_cfg['name']} ---")

    # Load model
    model, tokenizer = load_model_with_rope(config_path, rope_cfg["name"])

    # Extract attention
    tokens, attentions = extract_attention_from_model(
        model,
        tokenizer,
        input_text,
        max_length=config["visualization"]["max_seq_length"],
    )

    # Save heatmaps
    save_dir = f"../results/{rope_cfg['name']}"
    plot_heatmaps_with_bertviz(
        tokens,
        attentions,
        config["visualization"]["layers_to_plot"],
        config["visualization"]["heads_to_plot"],
        save_dir,
        rope_cfg["name"],
    )

    # Clear GPU memory
    del model
    torch.cuda.empty_cache()
