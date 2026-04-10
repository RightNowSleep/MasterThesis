"""
eval/plot_entropy_fig3.py
--------------------
Generate a single figure (modified Fig 3) from a JSON file produced by EntropyEvaluator.

Usage
-----
    python eval/plot_entropy_fig3.py --input results/entropy/llama-7b_none.json

Figure produced
---------------
A single figure showing entropy vs sequence length for multiple layers.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Sequence

import matplotlib

# 设置中文字体
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Global aesthetics
# ---------------------------------------------------------------------------

# Fixed colours for the six experiment sequence lengths
_LENGTH_COLORS: dict[int, str] = {
    512: "#E65100",  # deep orange
    1024: "#2E7D32",  # dark green
    1536: "#1976D2",  # medium blue
    2048: "#B71C1C",  # dark red
    2560: "#7B1FA2",  # purple
    3072: "#6A1B9A",  # deep purple
}
_FALLBACK_CMAP = plt.cm.tab10  # for unexpected lengths

# Figure size (width, height) in inches – optimized for paper
_FIG_SIZE = (10, 6)  # Suitable for paper
_DPI = 300
_TITLE_FS = 16
_LABEL_FS = 14
_TICK_FS = 12
_LEGEND_FS = 12

# Target lengths to include (6 lengths)
_TARGET_LENGTHS = [512, 1024, 1536, 2048, 2560, 3072]

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _get_color(seq_len: int, lengths: Sequence[int]) -> str:
    """
    Return a consistent colour for a given sequence length.

    Uses a fixed palette (_LENGTH_COLORS) for known lengths; falls back to
    tab10 colormap for unexpected values.

    Args:
        seq_len: The sequence length to look up.
        lengths: All available sequence lengths (used for fallback indexing).

    Returns:
        A matplotlib-compatible colour string (e.g., '#E65100').
    """
    if seq_len in _LENGTH_COLORS:
        return _LENGTH_COLORS[seq_len]
    idx = list(lengths).index(seq_len)
    return _FALLBACK_CMAP(idx / max(len(lengths) - 1, 1))


def _savefig(
    fig: plt.Figure,
    path: str,
    dpi: int = _DPI,
    constrained_layout: bool = False,
) -> None:
    if not constrained_layout:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def _end_label(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    label: str,
    color: str,
    fs: int = 12,
) -> None:
    """
    Annotate the last point of a line with its label.

    Places a bold text label to the right of the final (x, y) coordinate
    on the given axes, used for line-plot legends.

    Args:
        ax: Matplotlib Axes to annotate on.
        xs: X-coordinate array.
        ys: Y-coordinate array.
        label: Text label to display.
        colour: Text colour.
        fs: Font size, defaults to 12.

    Returns:
        None.
    """
    ax.annotate(
        label,
        xy=(xs[-1], ys[-1]),
        xytext=(5, 0),
        textcoords="offset points",
        color=color,
        fontsize=fs,
        va="center",
        fontweight="bold",
    )


def _norm_from_raw_np(arr: np.ndarray) -> np.ndarray:
    """
    Compute H_norm = H / log(t+1) along the last axis of a numpy array.

    Args:
        arr: ndarray of shape [..., T].

    Returns:
        ndarray of shape [..., T] with values in [0, 1].
    """
    T = arr.shape[-1]
    pos = np.arange(T, dtype=np.float64)
    den = np.log(pos + 1.0).clip(min=1e-6)
    return arr / den


def _normalize_data(data: dict) -> dict:
    """
    Convert the new single-length JSON format produced by the updated EntropyEvaluator
    into the legacy multi-length format expected by all plotting functions.

    New format:
        data["max_length"] = int,
        data["results"] = {metric_key: value, ...}

    Legacy format:
        data["lengths"] = [int, ...]
        data["results"] = {str(seq_len): {metric_key: value, ...}}

    If the data already uses the legacy format (has a "lengths" key) it is
    returned unchanged so this function is safe to call unconditionally.

    For single-length data, shorter lengths are derived by truncating the
    primary [L, H, T] arrays and recomputing derived metrics.

    Args:
        data: The data dictionary to normalize.

    Returns:
        The normalized data dictionary.
    """
    if "lengths" in data:
        return data

    max_len = data["max_length"]
    res = data["results"]
    target_lengths = [sl for sl in _TARGET_LENGTHS if sl <= max_len]

    if not target_lengths:
        target_lengths = [max_len]

    primary_raw = np.array(res["entropy_head_layer_position"])  # [L, H, T]
    primary_norm = np.array(res["norm_entropy_head_layer_position"])  # [L, H, T]
    L, H, _ = primary_raw.shape

    new_results = {}

    for sl in target_lengths:
        sl_res = {}

        sl_res["entropy_head_layer_position"] = primary_raw[:, :, :sl].round(5).tolist()
        sl_res["norm_entropy_head_layer_position"] = (
            primary_norm[:, :, :sl].round(5).tolist()
        )

        sl_res["entropy_layer_position"] = (
            primary_raw[:, :, :sl].mean(axis=1).round(5).tolist()
        )
        sl_res["norm_entropy_layer_position"] = (
            primary_norm[:, :, :sl].mean(axis=1).round(5).tolist()
        )

        sl_res["entropy_head_layer"] = (
            primary_raw[:, :, :sl].mean(axis=2).round(5).tolist()
        )
        sl_res["norm_entropy_head_layer"] = (
            primary_norm[:, :, :sl].mean(axis=2).round(5).tolist()
        )

        norm_per_head_mean = primary_norm[:, :, :sl].mean(axis=2)  # [L, H]
        sl_res["head_norm_std_by_layer"] = np.round(
            norm_per_head_mean.std(axis=1),
            5,
        ).tolist()

        raw_flat = primary_raw[:, :, :sl].reshape(L, -1)  # [L, H*sl]
        norm_flat = primary_norm[:, :, :sl].reshape(L, -1)
        raw_q, nrm_q = [], []
        for l in range(L):
            raw_q.append(
                np.round(
                    np.quantile(raw_flat[l], [0.0, 0.25, 0.5, 0.75, 1.0]),
                    5,
                ).tolist()
            )
            nrm_q.append(
                np.round(
                    np.quantile(norm_flat[l], [0.0, 0.25, 0.5, 0.75, 1.0]),
                    5,
                ).tolist()
            )
        sl_res["raw_entropy_quartiles_by_layer"] = raw_q
        sl_res["norm_entropy_quartiles_by_layer"] = nrm_q

        sl_res["top_k_concentration"] = np.round(
            np.array(res["top_k_concentration"])[:sl],
            5,
        ).tolist()
        sl_res["top_k_boundary"] = min(res["top_k_boundary"], sl - 1)

        new_results[str(sl)] = sl_res

    data["lengths"] = target_lengths
    data["results"] = new_results
    return data


def _auto_select_layers(data: dict, n: int = 4) -> List[int]:
    """
    Return the indices of the *n* layers with the highest average head-entropy
    standard deviation across all evaluated sequence lengths.

    These layers show the most intra-layer head specialisation and are the
    most informative to inspect in detail.

    Args:
        data: The data dictionary containing entropy results.
        n: Number of layers to select, default is 4.

    Returns:
        List of layer indices sorted by descending standard deviation.
    """
    lengths = data["lengths"]
    num_layers = data["num_layers"]
    avg_std = np.zeros(num_layers)
    for sl in lengths:
        std = np.array(data["results"][str(sl)]["head_norm_std_by_layer"])
        avg_std += std
    avg_std /= len(lengths)
    selected = np.argsort(avg_std)[-n:][::-1].tolist()
    return selected  # descending std


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------


def plot_entropy_vs_position(
    data: dict,
    out_dir: str,
    fmt: str,
    dpi: int,
    selected_layers: List[int] | None = None,
    method_name: str = "",
) -> None:
    """
    Generate a single figure showing entropy vs token position for multiple layers.

    Args:
        data: The data dictionary containing entropy results.
        out_dir: Output directory for saving figures.
        fmt: Output format (e.g., 'png', 'pdf').
        dpi: Resolution for raster formats.
        selected_layers: Optional list of layer indices to plot.
            If None, auto-selects layers with highest head-entropy std.
        method_name: Prefix for output filenames.
    """
    # Auto-select up to 8 layers if not specified
    if selected_layers is None:
        # Get up to 8 layers, but no more than available layers
        num_layers = min(data["num_layers"], 8)
        selected_layers = _auto_select_layers(data, n=num_layers)
    else:
        # Ensure we don't have more than 8 layers
        selected_layers = selected_layers[:8]

    print(f"  Selected layers: {selected_layers}")

    lengths = data["lengths"]
    # Use the longest sequence length for plotting
    max_length = max(lengths)

    # Create a single figure (English version)
    fig, ax = plt.subplots(figsize=_FIG_SIZE)

    # Plot each layer as a separate curve
    for layer_idx in selected_layers:
        # Get entropy data for the longest sequence length
        matrix = np.array(
            data["results"][str(max_length)]["entropy_layer_position"]
        )  # [L, T] - using raw entropy
        # Get the entropy values for this layer
        entropy_values = matrix[layer_idx]  # [T]
        # Create position array
        positions = np.arange(len(entropy_values))

        # Get color for this layer
        color = _get_color(layer_idx, selected_layers)

        # Plot the curve with all points
        ax.plot(
            positions,
            entropy_values,
            color=color,
            linewidth=1.5,
            marker=None,
            label=f"Layer {layer_idx}",
            linestyle="-",
        )

        # Add end label
        _end_label(ax, positions, entropy_values, f"Layer {layer_idx}", color)

    # Add vertical line at position 2048 for easy identification
    if 2048 < max_length:
        ax.axvline(x=2048, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
        # Add horizontal text label
        ax.text(
            2048,
            ax.get_ylim()[0] + 0.1,
            "Position 2048",
            ha="center",
            rotation=0,
            fontsize=_TICK_FS,
        )

    # Set labels and title
    ax.set_xlabel("Token Position", fontsize=_LABEL_FS)
    ax.set_ylabel("Entropy (nats)", fontsize=_LABEL_FS)
    ax.set_title(
        "Entropy vs Token Position for Different Layers",
        fontsize=_TITLE_FS,
        pad=10,
    )

    # Set tick parameters
    ax.tick_params(labelsize=_TICK_FS)

    # Add grid
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    # Add legend
    ax.legend(fontsize=_LEGEND_FS, framealpha=0.8, loc="best")

    # Save the figure with the same naming convention as the original plot_entropy.py
    _savefig(
        fig,
        os.path.join(out_dir, f"{method_name}_fig03_entropy_vs_position_raw.{fmt}"),
        dpi,
    )

    # Create Chinese version
    fig_cn, ax_cn = plt.subplots(figsize=_FIG_SIZE)

    # Plot each layer as a separate curve
    for layer_idx in selected_layers:
        # Get entropy data for the longest sequence length
        matrix = np.array(
            data["results"][str(max_length)]["entropy_layer_position"]
        )  # [L, T] - using raw entropy
        # Get the entropy values for this layer
        entropy_values = matrix[layer_idx]  # [T]
        # Create position array
        positions = np.arange(len(entropy_values))

        # Get color for this layer
        color = _get_color(layer_idx, selected_layers)

        # Plot the curve with all points
        ax_cn.plot(
            positions,
            entropy_values,
            color=color,
            linewidth=1.5,
            marker=None,
            label=f"Layer {layer_idx}",
            linestyle="-",
        )

        # Add end label
        _end_label(ax_cn, positions, entropy_values, f"Layer {layer_idx}", color)

    # Add vertical line at position 2048 for easy identification
    if 2048 < max_length:
        ax_cn.axvline(x=2048, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
        # Add horizontal text label
        ax_cn.text(
            2048,
            ax_cn.get_ylim()[0] + 0.1,
            "Token索引 2048",
            ha="center",
            rotation=0,
            fontsize=_TICK_FS,
        )

    # Set labels (Chinese)
    ax_cn.set_xlabel("Token索引", fontsize=_LABEL_FS)
    ax_cn.set_ylabel("熵值 (nats)", fontsize=_LABEL_FS)
    # No title for Chinese version

    # Set tick parameters
    ax_cn.tick_params(labelsize=_TICK_FS)

    # Add grid
    ax_cn.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    # Add legend
    ax_cn.legend(fontsize=_LEGEND_FS, framealpha=0.8, loc="best")

    # Save the Chinese version
    _savefig(
        fig_cn,
        os.path.join(out_dir, f"{method_name}_fig03_entropy_vs_position_raw_cn.{fmt}"),
        dpi,
    )


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Plot entropy vs sequence length (modified Fig 3)"
    )
    parser.add_argument(
        "--input",
        default="results/entropy/llama-7b_inverse-dual-rope_dynamic.json",
        help="Path to the JSON file produced by EntropyEvaluator",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for saving figures. Default: <input_dir>/plots/<input_stem>",
    )
    parser.add_argument(
        "--fmt",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=_DPI,
        help="Resolution for raster formats",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        help="Layer indices to plot (default: auto-select up to 6 layers)",
    )
    parser.add_argument(
        "--method-name",
        default="",
        help="Prefix for output filenames",
    )
    args = parser.parse_args()

    # Load and normalize data
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    # Extract method name from input filename if not provided
    if not args.method_name:
        stem = input_path.stem
        args.method_name = stem.split("_", 1)[1] if "_" in stem else stem

    # Determine output directory
    out_dir = args.out_dir or str(input_path.parent / "plots" / input_path.stem)
    os.makedirs(out_dir, exist_ok=True)

    with open(input_path, "r") as f:
        data = json.load(f)
    data = _normalize_data(data)

    # Plot the figure
    plot_entropy_vs_position(
        data=data,
        out_dir=out_dir,
        fmt=args.fmt,
        dpi=args.dpi,
        selected_layers=args.layers,
        method_name=args.method_name,
    )


if __name__ == "__main__":
    main()
