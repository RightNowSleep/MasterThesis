import os
import sys
import argparse
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from models.pe_llama import *

ROPE_TYPE_TO_CLASS = {
    "rope": LlamaRotaryEmbedding,
    "linear": LlamaLinearScalingRotaryEmbedding,
    "ntk": LlamaNTKAwareScaledRotaryEmbedding,
    "part-ntk": LlamaNTKByPartsScaledRotaryEmbedding,
    "yarn": LlamaYarnScaledRotaryEmbedding,
    "my-rope": LlamaMyRotaryEmbedding,
    "my-rope-scaled": LlamaMyScaledRotaryEmbedding,
    "my-rope2": LlamaMyRotaryEmbedding2,
    "my-rope2-scaled": LlamaMyScaledRotaryEmbedding2,
    "block-layered": LlamaBlockLayeredRotaryEmbedding,
    "block-layered-scaled": LlamaBlockLayeredScaledRotaryEmbedding,
    "freq-smooth": LlamaFreqSmoothRotaryEmbedding,
    "freq-smooth-scaled": LlamaFreqSmoothScaledRotaryEmbedding,
    "freq-reciprocal": LlamaFreqReciprocalRotaryEmbedding,
    "freq-reciprocal-scaled": LlamaFreqReciprocalScaledRotaryEmbedding,
}


def parse_args():
    """Parse command-line arguments for RoPE position heatmap visualization.

    Returns:
        argparse.Namespace: Parsed arguments containing visualization parameters.
    """
    parser = argparse.ArgumentParser(
        description="RoPE Position Heatmap Visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["rope", "linear", "ntk", "part-ntk", "freq-reciprocal"],
        choices=list(ROPE_TYPE_TO_CLASS.keys()),
        help="RoPE methods to visualize (space-separated)",
    )

    parser.add_argument("--dim", type=int, default=128, help="RoPE dimension")
    parser.add_argument("--base", type=int, default=100, help="RoPE base frequency")
    parser.add_argument(
        "--original-L",
        type=int,
        default=64,
        help="Original max position embeddings",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length for visualization",
    )
    parser.add_argument(
        "--scaling-factor",
        type=float,
        default=8.0,
        help="Scaling factor S",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Alpha for NTK-by-parts / YaRN",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=32.0,
        help="Beta for NTK-by-parts / YaRN",
    )
    parser.add_argument("--layer-idx", type=int, default=0, help="Layer index")
    parser.add_argument(
        "--num-layers",
        type=int,
        default=32,
        help="Number of hidden layers",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        default=False,
        help="Use dynamic scaling for methods that support it",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="drawer/position_heatmap",
        help="Output directory and base filename (method name will be prepended)",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=4.5,
        help="Figure width in inches (3.5 for single column, 7 for double column)",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=6.0,
        help="Figure height in inches",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="DPI for output image (300 for screen, 600 for print)",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="plasma",
        help="Colormap name (plasma, inferno, viridis recommended for papers)",
    )

    return parser.parse_args()


def create_rope_model(method, args, device):
    """Create a RoPE model instance based on method name and arguments.

    Args:
        method: RoPE method name (e.g., 'rope', 'linear', 'ntk').
        args: Parsed arguments containing model configuration.
        device: Torch device to place the model on.

    Returns:
        RoPE embedding model instance configured with the specified parameters.
    """
    cls = ROPE_TYPE_TO_CLASS[method]

    if method == "rope":
        max_pos = args.seq_len
    else:
        max_pos = int(args.original_L * args.scaling_factor)

    common_kwargs = {
        "dim": args.dim,
        "max_position_embeddings": max_pos,
        "base": args.base,
        "device": device,
    }

    if method == "rope":
        return cls(**common_kwargs)

    extra_kwargs = {
        "scaling_factor": args.scaling_factor,
        "original_max_position_embeddings": args.original_L,
        "dynamic": getattr(args, "dynamic", False),
    }

    if method in ["part-ntk", "yarn"]:
        extra_kwargs["alpha"] = args.alpha
        extra_kwargs["beta"] = args.beta

    needs_layer_params = (
        method.startswith("my-rope")
        or method.startswith("block-layered")
        or method.startswith("freq-smooth")
        or method.startswith("freq-reciprocal")
    )
    if needs_layer_params:
        extra_kwargs["layer_idx"] = args.layer_idx
        extra_kwargs["num_hidden_layers"] = args.num_layers

    return cls(**common_kwargs, **extra_kwargs)


def generate_title(method, args):
    """Generate a formatted subplot title for a RoPE method.

    Args:
        method: RoPE method name.
        args: Parsed arguments containing scaling factor and dynamic flag.

    Returns:
        Formatted title string with method name and scaling information.
    """
    name_map = {
        "rope": "Standard RoPE",
        "linear": "Linear Scaling (PI)",
        "ntk": "NTK-aware",
        "part-ntk": "NTK-by-parts",
        "yarn": "YaRN",
        "my-rope": "My-RoPE",
        "my-rope-scaled": "My-RoPE (scaled)",
        "my-rope2": "My-RoPE2",
        "my-rope2-scaled": "My-RoPE2 (scaled)",
        "block-layered": "Block-Layered",
        "block-layered-scaled": "Block-Layered (scaled)",
        "freq-smooth": "Freq-Smooth",
        "freq-smooth-scaled": "Freq-Smooth (scaled)",
        "freq-reciprocal": "Freq-Reciprocal",
        "freq-reciprocal-scaled": "Freq-Reciprocal (scaled)",
    }
    title = name_map.get(method, method)
    if method != "rope":
        title += f"\nS={args.scaling_factor}"
        if getattr(args, "dynamic", False):
            title += " (dynamic)"
    return title


def draw_original_length_box(ax, original_L, dim_half, fontsize=10):
    """Draw a dashed rectangle to highlight the original length region on heatmap.

    Args:
        ax: Matplotlib axes object to draw on.
        original_L: Original maximum position embedding length.
        dim_half: Half of the RoPE dimension (number of frequency pairs).
        fontsize: Font size for the annotation text.
    """
    rect = plt.Rectangle(
        (0, 0),
        dim_half - 1,
        original_L - 1,
        fill=False,
        edgecolor="white",
        linestyle="--",
        linewidth=1.5,
        alpha=0.9,
    )
    ax.add_patch(rect)

    text_x = dim_half * 0.85
    text_y = original_L - 1

    ax.text(
        text_x,
        text_y,
        f"Original L={original_L}",
        color="white",
        fontsize=fontsize,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="black",
            edgecolor="white",
            alpha=0.7,
            linewidth=0.5,
        ),
    )


def main():
    """Main function to generate and save RoPE position heatmaps.

    Parses command-line arguments, creates RoPE models for each specified method,
    generates frequency heatmaps, and saves them to the output directory.
    """
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    output_dir = args.output
    base_filename = os.path.basename(args.output)
    _, ext = os.path.splitext(base_filename)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 100,
        }
    )

    for method in args.methods:
        model = create_rope_model(method, args, device)
        if method == "rope":
            actual_seq_len = args.seq_len
        else:
            actual_seq_len = int(args.seq_len * args.scaling_factor)
        model._set_cos_sin_cache(actual_seq_len, device=device, dtype=dtype)
        freq = model.cos_cached.cpu().numpy()
        title = generate_title(method, args)

        dim_half = args.dim // 2

        fig, ax = plt.subplots(figsize=(12, 12))

        im = ax.imshow(
            freq,
            cmap=args.cmap,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
        )

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Dimension $i$", fontsize=11)
        ax.set_ylabel("Position", fontsize=11)
        ax.set_xlim(0, dim_half - 1)
        ax.set_ylim(0, actual_seq_len - 1)

        n_xticks = min(6, dim_half)
        xtick_positions = np.linspace(0, dim_half - 1, n_xticks, dtype=int)
        ax.set_xticks(xtick_positions)

        n_yticks = min(8, actual_seq_len)
        ytick_positions = np.linspace(0, actual_seq_len - 1, n_yticks, dtype=int)
        ax.set_yticks(ytick_positions)

        ax.grid(True, alpha=0.2, linestyle=":", color="white", linewidth=0.5)

        if method == "rope":
            original_L_display = args.original_L
        else:
            original_L_display = int(args.original_L * args.scaling_factor)

        if original_L_display < actual_seq_len:
            draw_original_length_box(ax, original_L_display, dim_half, fontsize=9)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, aspect=30)
        cbar.set_label("Frequency Value", rotation=270, labelpad=15, fontsize=11)
        cbar.ax.tick_params(labelsize=9)

        plt.tight_layout()

        method_filename = f"{method}{ext}"
        output_path = os.path.join(output_dir, method_filename)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
