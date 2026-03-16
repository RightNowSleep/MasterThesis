import os
import sys
import argparse
import math
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
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


def get_subplot_layout(n):
    """Intelligently compute subplot layout, return (nrows, ncols)."""
    if n <= 0:
        return 1, 1
    elif n == 1:
        return 1, 1
    elif n == 2:
        return 1, 2
    elif n == 3:
        return 1, 3
    elif n == 4:
        return 2, 2
    elif n <= 6:
        return 2, 3
    elif n <= 9:
        return 3, 3
    elif n <= 12:
        return 3, 4
    elif n <= 16:
        return 4, 4
    elif n <= 20:
        return 4, 5
    elif n <= 25:
        return 5, 5
    else:
        ncols = int(math.ceil(math.sqrt(n)))
        nrows = int(math.ceil(n / ncols))
        return nrows, ncols


def parse_args():
    parser = argparse.ArgumentParser(
        description="RoPE Position Heatmap Visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["rope", "freq-reciprocal"],
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
        "--output",
        type=str,
        default="drawer/rope_heatmap.png",
        help="Output file path",
    )
    parser.add_argument(
        "--fig-width",
        type=int,
        default=18,
        help="Figure width (inches)",
    )
    parser.add_argument(
        "--fig-height",
        type=int,
        default=8,
        help="Figure height (inches)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for output image",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Colormap name",
    )
    parser.add_argument(
        "--share-colorbar",
        action="store_true",
        default=True,
        help="Share a single colorbar across all subplots",
    )
    parser.add_argument("--title", type=str, default=None, help="Custom figure title")

    parser.add_argument(
        "--show-mapping",
        action="store_true",
        default=True,
        help="Show dimension mapping from RoPE to FreqReciprocal with dashed lines",
    )
    parser.add_argument(
        "--mapping-block-sizes",
        type=float,
        nargs="+",
        default=[2.0, 4.0, 6.0, 8.0],
        help="Block sizes to visualize mapping (used with --show-mapping)",
    )

    return parser.parse_args()


def create_rope_model(method, args, device):
    """Create a RoPE model based on method name and arguments."""
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
    """Generate subplot title."""
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


def find_dimension_for_block_size(model, target_block_size):
    """
    Find the dimension index i where block_size is closest to target_block_size.
    Returns the dimension index and the actual block_size at that dimension.
    """
    if not hasattr(model, "block_sizes"):
        return None, None
    block_sizes = model.block_sizes.cpu().numpy()
    idx = np.argmin(np.abs(block_sizes - target_block_size))
    return idx, block_sizes[idx]


def draw_mapping_lines(axes, methods, freq_matrices, args, models):
    """
    Draw dashed lines on heatmaps to show RoPE -> FreqReciprocal dimension mapping.
    """
    if not args.show_mapping:
        return
    if len(methods) != 2 or "rope" not in methods or "freq-reciprocal" not in methods:
        print(
            "Warning: --show-mapping requires exactly 'rope' and 'freq-reciprocal' methods"
        )
        return

    rope_idx = methods.index("rope")
    freq_recip_idx = methods.index("freq-reciprocal")
    freq_recip_model = models[freq_recip_idx]

    seq_len = args.seq_len
    rope_seq_len = freq_matrices[rope_idx].shape[0]
    freq_recip_seq_len = freq_matrices[freq_recip_idx].shape[0]

    colors = ["#E63946", "#1D3557", "#2A9D8F", "#F4A261"]
    linestyle = "--"

    legend_handles = []

    for i, target_b in enumerate(args.mapping_block_sizes):
        dim_idx, actual_b = find_dimension_for_block_size(freq_recip_model, target_b)
        if dim_idx is None:
            continue

        color = colors[i % len(colors)]

        rope_line_len = rope_seq_len
        freq_recip_line_len = int(seq_len * target_b)

        axes[rope_idx].axvline(
            x=dim_idx,
            ymin=0,
            ymax=1,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            alpha=0.9,
        )

        axes[freq_recip_idx].axvline(
            x=dim_idx,
            ymin=0,
            ymax=freq_recip_line_len / freq_recip_seq_len,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            alpha=0.9,
        )

        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=color,
                linestyle=linestyle,
                linewidth=2,
                label=f"b={target_b:.1f}, i={dim_idx}",
            )
        )

    if legend_handles:
        axes[freq_recip_idx].legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=8,
            framealpha=0.9,
        )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    n = len(args.methods)
    nrows, ncols = get_subplot_layout(n)

    freq_matrices = []
    titles = []
    seq_lens = []
    models = []

    for method in args.methods:
        model = create_rope_model(method, args, device)
        if method == "rope":
            actual_seq_len = args.seq_len
        else:
            actual_seq_len = int(args.seq_len * args.scaling_factor)
        model._set_cos_sin_cache(actual_seq_len, device=device, dtype=dtype)
        freq = model.cos_cached.cpu().numpy()
        freq_matrices.append(freq)
        titles.append(generate_title(method, args))
        seq_lens.append(actual_seq_len)
        models.append(model)

    if args.share_colorbar:
        vmin = min(f.min() for f in freq_matrices)
        vmax = max(f.max() for f in freq_matrices)
        norm = Normalize(vmin=vmin, vmax=vmax)

    fig_width = args.fig_width if ncols <= 2 else args.fig_width + (ncols - 2) * 4
    fig_height = args.fig_height if nrows == 1 else args.fig_height + (nrows - 1) * 3

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
    )

    if n == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()

    main_title = (
        args.title or f"RoPE Frequency Heatmap (dim={args.dim}, L={args.original_L})"
    )
    fig.suptitle(main_title, fontsize=16)

    images = []
    for i, (freq, title) in enumerate(zip(freq_matrices, titles)):
        if args.share_colorbar:
            im = axes[i].imshow(freq, cmap=args.cmap, norm=norm, aspect="auto")
        else:
            im = axes[i].imshow(freq, cmap=args.cmap, aspect="auto")
        images.append(im)
        axes[i].set_title(title, fontsize=10)
        axes[i].set_xlabel(f"Dimension i (0-{args.dim//2 - 1})")
        axes[i].set_ylabel(f"Position (0-{seq_lens[i] - 1})")
        axes[i].set_xlim(0, args.dim // 2 - 1)
        axes[i].set_ylim(0, seq_lens[i] - 1)

    for i in range(n, len(axes)):
        axes[i].axis("off")

    draw_mapping_lines(axes, args.methods, freq_matrices, args, models)

    if args.share_colorbar:
        visible_axes = [axes[i] for i in range(n)]
        cbar = fig.colorbar(images[-1], ax=visible_axes, shrink=0.8, pad=0.02)
        cbar.set_label("Frequency Value", rotation=270, labelpad=20, fontsize=12)
    else:
        for i in range(n):
            fig.colorbar(images[i], ax=axes[i], shrink=0.8)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
