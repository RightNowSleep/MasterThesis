import matplotlib.pyplot as plt
import seaborn as sns
from bertviz import head_view, model_view
import os


def plot_heatmaps_with_bertviz(
    tokens,
    attentions,
    layer_indices,
    head_indices,
    save_dir,
    model_name,
):
    """
    Generate interactive heatmaps using BertViz and save as HTML.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Global model view (Model View)
    html_path = os.path.join(save_dir, f"{model_name}_model_view.html")
    model_view(attentions, tokens, html_action="save", save_path=html_path)
    print(f"Model view saved to {html_path}")

    # 2. Specific layer and head view (Head View)
    for layer in layer_indices:
        for head in head_indices:
            html_path = os.path.join(
                save_dir,
                f"{model_name}_layer{layer}_head{head}.html",
            )
            head_view(
                attentions[layer],
                tokens,
                layer,
                head,
                html_action="save",
                save_path=html_path,
            )
            print(f"Head view saved to {html_path}")


def plot_static_heatmap(tokens, attention_matrix, title, save_path):
    """
    Generate static heatmap using matplotlib/seaborn.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_matrix, xticklabels=tokens, yticklabels=tokens, cmap="viridis"
    )
    plt.title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
