from src.model_loader import load_model_with_rope
from src.attention_extractor import extract_attention_from_model
from src.visualizer import plot_heatmaps_with_bertviz, plot_static_heatmap
import yaml
import torch

# 1. 加载配置
config_path = "../config/model_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# 2. 准备输入数据
with open("../data/your_dataset.txt", 'r') as f:
    input_text = f.read().strip()

# 3. 遍历所有RoPE配置
for rope_cfg in config['rope_configs']:
    print(f"\n--- Analyzing {rope_cfg['name']} ---")

    # 加载模型
    model, tokenizer = load_model_with_rope(config_path, rope_cfg['name'])

    # 提取注意力
    tokens, attentions = extract_attention_from_model(
        model,
        tokenizer,
        input_text,
        max_length=config['visualization']['max_seq_length']
    )

    # 保存热力图
    save_dir = f"../results/{rope_cfg['name']}"
    plot_heatmaps_with_bertviz(
        tokens,
        attentions,
        config['visualization']['layers_to_plot'],
        config['visualization']['heads_to_plot'],
        save_dir,
        rope_cfg['name']
    )

    # 清理显存
    del model
    torch.cuda.empty_cache()