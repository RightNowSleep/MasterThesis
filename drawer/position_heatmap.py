"""RoPE position embedding heatmap visualization for context extension analysis.

This module generates 2D heatmap visualizations of Rotary Position Embedding (RoPE)
cosine frequency values across position indices (y-axis) and embedding dimensions
(x-axis) for various RoPE extension methods.

The heatmaps reveal how different scaling strategies distribute positional
information across the extended context window:

    - Standard RoPE: Shows regular sinusoidal patterns within the original context.
    - Linear Scaling (PI): Uniformly compresses all frequencies.
    - NTK-aware: Applies Neural Tangent Kernel-inspired frequency scaling.
    - NTK-by-Parts: Uses band-dependent scaling with smooth transitions.
    - YaRN: YaRN scaling with alpha/beta parameters.
    - My-RoPE / My-RoPE2: Custom layer-aware scaling methods.
    - Block-Layered: Block-wise layered scaling approach.
    - Freq-Smooth / Freq-Reciprocal: Frequency-domain smoothing or reciprocal scaling.

Each heatmap includes an optional bounding box highlighting the original training
context length region, making it easy to identify which dimensions preserve
resolution in-distribution vs. out-of-distribution areas.

Usage:
    python drawer/position_heatmap.py --methods rope linear ntk freq-reciprocal --seq-len 256

Output:
    drawer/position_heatmap/{method}.png: Individual heatmap per method at configured DPI.
"""

import os
import sys
import argparse
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from models.pe_llama import *

# Mapping from CLI method names to RoPE implementation classes
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

    Defines all configurable parameters for the visualization including RoPE method
    selection, model hyperparameters, output settings, and plot styling options.

    Returns:
        argparse.Namespace: Parsed namespace containing the following attributes:
            methods (List[str]): List of RoPE method names to visualize.
            dim (int): RoPE dimension size (must be even).
            base (int): Base frequency for inverse frequency computation.
            original_L (int): Original maximum position embeddings (training length).
            seq_len (int): Sequence length for visualization (positions on y-axis).
            scaling_factor (float): Context extension ratio S.
            alpha (float): Lower threshold for NTK-by-Parts / YaRN methods.
            beta (float): Upper threshold for NTK-by-Parts / YaRN methods.
            layer_idx (int): Layer index for layer-aware methods.
            num_layers (int): Total number of hidden layers for normalization.
            dynamic (bool): Whether to use dynamic scaling mode.
            output (str): Output directory and base filename template.
            fig_width (float): Figure width in inches.
            fig_height (float): Figure height in inches.
            dpi (int): Output image resolution in dots per inch.
            cmap (str): Matplotlib colormap name for heatmaps.
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
    """Create a RoPE model instance based on method name and configuration arguments.

    Constructs the appropriate RoPE embedding class with method-specific parameters.
    Handles the different constructor signatures across RoPE variants by building
    separate keyword argument dictionaries for common and extra parameters.

    Args:
        method (str): RoPE method identifier (e.g., 'rope', 'linear', 'ntk',
            'part-ntk', 'freq-reciprocal'). Must be a key in ROPE_TYPE_TO_CLASS.
        args (argparse.Namespace): Parsed command-line arguments containing model
            configuration parameters (dim, base, scaling_factor, etc.).
        device (torch.device): Torch device (CPU or CUDA) to place model tensors on.

    Returns:
        An instantiated RoPE embedding model object configured with the specified
        parameters and ready for cosine/sine cache generation.
    """
    cls = ROPE_TYPE_TO_CLASS[method]

    # Standard RoPE uses the raw sequence length; others use scaled max positions
    if method == "rope":
        max_pos = args.seq_len
    else:
        max_pos = int(args.original_L * args.scaling_factor)

    # Common parameters shared by all RoPE variants
    common_kwargs = {
        "dim": args.dim,
        "max_position_embeddings": max_pos,
        "base": args.base,
        "device": device,
    }

    if method == "rope":
        return cls(**common_kwargs)

    # Extra parameters required by scaled/extended RoPE variants
    extra_kwargs = {
        "scaling_factor": args.scaling_factor,
        "original_max_position_embeddings": args.original_L,
        "dynamic": getattr(args, "dynamic", False),
    }

    # NTK-by-Parts and YaRN require additional alpha/beta thresholds
    if method in ["part-ntk", "yarn"]:
        extra_kwargs["alpha"] = args.alpha
        extra_kwargs["beta"] = args.beta

    # Layer-aware methods (My-RoPE, Block-Layered, Freq-Smooth, Freq-Reciprocal)
    # require layer index and total layer count for normalization
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
    """Generate a formatted subplot title string for a RoPE method.

    Maps internal method identifiers to human-readable display names and appends
    scaling information (extension ratio S and dynamic flag) for non-baseline methods.

    Args:
        method (str): Internal RoPE method identifier string.
        args (argparse.Namespace): Parsed arguments containing scaling_factor and
            dynamic attributes for title decoration.

    Returns:
        str: Formatted multi-line title string suitable for matplotlib subplot titles.
            Includes method name, scaling factor, and dynamic mode indicator where applicable.
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
    """Draw a dashed rectangle highlighting the original training context region.

    Renders a white dashed bounding box over the portion of the heatmap that
    corresponds to positions and dimensions within the original (non-extended)
    training context length. This visual guide helps distinguish in-distribution
    from out-of-distribution regions in the frequency pattern.

    Args:
        ax (matplotlib.axes.Axes): Matplotlib axes object to draw the annotation on.
        original_L (int): Original maximum position embedding length defining the
            vertical extent of the highlighted region.
        dim_half (int): Half of the RoPE dimension (number of frequency pairs),
            defining the horizontal extent of the highlighted region.
        fontsize (int, optional): Font size for the annotation label text.
            Defaults to 10.

    Returns:
        None
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

    # Position annotation text near the top-right corner of the box
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
    """Main entry point for RoPE position heatmap generation.

    Orchestrates the full visualization pipeline:

        1. Parse command-line arguments for configuration.
        2. Detect available compute device (CUDA fallback to CPU).
        3. Configure matplotlib publication-quality style settings.
        4. For each requested RoPE method:
           a. Instantiate the RoPE model with configured parameters.
           b. Generate cosine frequency cache for the target sequence length.
           c. Create a 2D heatmap visualization of cos_cached values.
           d. Overlay the original context length bounding box if applicable.
           e. Save the figure to disk in the output directory.

    Returns:
        None
    """
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    output_dir = args.output
    base_filename = os.path.basename(args.output)
    _, ext = os.path.splitext(base_filename)

    # Apply publication-quality matplotlib style settings
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
        # Instantiate RoPE model and generate frequency cache
        model = create_rope_model(method, args, device)
        if method == "rope":
            actual_seq_len = args.seq_len
        else:
            actual_seq_len = int(args.seq_len * args.scaling_factor)
        model._set_cos_sin_cache(actual_seq_len, device=device, dtype=dtype)
        freq = model.cos_cached.cpu().numpy()
        title = generate_title(method, args)

        dim_half = args.dim // 2

        # Create square figure with heatmap
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

        # Set axis tick positions with sensible granularity
        n_xticks = min(6, dim_half)
        xtick_positions = np.linspace(0, dim_half - 1, n_xticks, dtype=int)
        ax.set_xticks(xtick_positions)

        n_yticks = min(8, actual_seq_len)
        ytick_positions = np.linspace(0, actual_seq_len - 1, n_yticks, dtype=int)
        ax.set_yticks(ytick_positions)

        # Subtle grid overlay for value reading
        ax.grid(True, alpha=0.2, linestyle=":", color="white", linewidth=0.5)

        # Determine whether to show the original context length bounding box
        if method == "rope":
            original_L_display = args.original_L
        else:
            original_L_display = int(args.original_L * args.scaling_factor)

        if original_L_display < actual_seq_len:
            draw_original_length_box(ax, original_L_display, dim_half, fontsize=9)

        # Add colorbar with descriptive label
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, aspect=30)
        cbar.set_label("Frequency Value", rotation=270, labelpad=15, fontsize=11)
        cbar.ax.tick_params(labelsize=9)

        plt.tight_layout()

        # Save individual heatmap file per method
        method_filename = f"{method}{ext}"
        output_path = os.path.join(output_dir, method_filename)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
