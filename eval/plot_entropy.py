"""
eval/plot_entropy.py
--------------------
Generate seven diagnostic figures from a JSON file produced by EntropyEvaluator.

Usage
-----
    python eval/plot_entropy.py --input results/entropy/llama-7b_none.json
    python eval/plot_entropy.py --input results/entropy/llama-7b_none.json \\
        --out-dir figures/entropy --fmt pdf --dpi 150

Seven figures
-------------
Fig 1  mean_entropy_by_layer_vs_length.png
       Line plot: x = layer index, y = mean entropy (nats).
       One coloured curve per evaluated sequence length.
       Reveals how entropy varies across the depth of the network at different contexts.

Fig 2  entropy_heatmap_head_layer.png
       Grid of heat-maps: one sub-plot per sequence length.
       x = head index, y = layer index, colour = mean entropy.
       Shows which (layer, head) pairs are sharp or diffuse.

Fig 3  entropy_vs_position.png
       Line plot: x = token position, y = mean entropy averaged over all layers and heads.
       One coloured curve per sequence length.
       Highlights how attention pattern sharpness changes along the sequence.

Fig 4  entropy_vs_seqlen.png
       Bar chart: x = sequence length, y = global mean entropy (mean over all layers,
       heads, and positions).
       Quick summary of how context length affects overall entropy.

Fig 5  entropy_boxplot_by_layer.png
       Box plots: one panel per sequence length.
       x = layer index, y = entropy distribution across heads × token positions × samples.
       Shows the spread/outliers in per-head entropy at each layer.

Fig 6  top_k_concentration_vs_position.png
       Line plot: x = token position, y = top-k attention mass fraction.
       One curve per sequence length.
       A low fraction → diffuse attention; high fraction → sharp / local attention.

Fig 7  delta_entropy_heatmap.png
       Heatmap: x = sequence length (excluding the baseline = min length),
                y = layer index,
                colour = Δmean_entropy relative to the shortest evaluated length.
       Reveals which layers are most affected by context extension.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; safe in headless environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Shared aesthetics
# ---------------------------------------------------------------------------

_PALETTE = plt.cm.viridis  # perceptually uniform; colour-blind friendly
_FIGSIZE = (10, 6)
_FIGSIZE_WIDE = (14, 5)
_TITLE_FS = 13
_LABEL_FS = 11
_TICK_FS = 9
_LEGEND_FS = 9


def _colour_cycle(n: int):
    """Return *n* evenly spaced colours from the palette."""
    return [_PALETTE(i / max(n - 1, 1)) for i in range(n)]


def _savefig(fig: plt.Figure, path: str, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ---------------------------------------------------------------------------
# Figure 1 – mean entropy by layer, one curve per length
# ---------------------------------------------------------------------------


def plot_mean_entropy_by_layer(
    data: dict,
    out_dir: str,
    fmt: str,
    dpi: int,
) -> None:
    """
    Fig 1: x = layer index, y = mean entropy, one line per seq-length.

    Interpretation
    ^^^^^^^^^^^^^^
    Early layers often show high entropy (broad attention over context);
    later layers typically sharpen.  Longer contexts can push entropy higher
    in all layers if the model spreads attention over more tokens.
    """
    lengths = data["lengths"]
    colours = _colour_cycle(len(lengths))

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for colour, length in zip(colours, lengths):
        me = data["results"][str(length)]["mean_entropy_by_layer"]
        layer_idx = list(range(len(me)))
        ax.plot(
            layer_idx,
            me,
            marker="o",
            markersize=3,
            label=f"len={length}",
            color=colour,
            linewidth=1.5,
        )

    ax.set_xlabel("Layer index", fontsize=_LABEL_FS)
    ax.set_ylabel("Mean attention entropy (nats)", fontsize=_LABEL_FS)
    ax.set_title("Mean attention entropy per layer", fontsize=_TITLE_FS)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=_LEGEND_FS, framealpha=0.7)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    _savefig(fig, os.path.join(out_dir, f"fig1_mean_entropy_by_layer.{fmt}"), dpi)


# ---------------------------------------------------------------------------
# Figure 2 – per-head entropy heatmap  (layers × heads)
# ---------------------------------------------------------------------------


def plot_entropy_heatmap_head_layer(
    data: dict,
    out_dir: str,
    fmt: str,
    dpi: int,
) -> None:
    """
    Fig 2: heat-map of mean entropy [layer × head], one sub-plot per seq-length.

    Interpretation
    ^^^^^^^^^^^^^^
    Bright cells = high entropy (diffuse / global attention heads).
    Dark cells   = low entropy  (sharp / local attention heads).
    Comparing sub-plots reveals which heads change behaviour with context length.
    """
    lengths = data["lengths"]
    n = len(lengths)
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 4.5 + 0.8, nrows * 3.5),
        squeeze=False,
    )

    vmin = vmax = None
    # compute global colour range for consistent scale across sub-plots
    all_vals = []
    for length in lengths:
        all_vals.extend(
            np.array(data["results"][str(length)]["mean_entropy_by_head"]).ravel()
        )
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)

    for ax_idx, length in enumerate(lengths):
        row, col = divmod(ax_idx, ncols)
        ax = axes[row][col]

        mat = np.array(data["results"][str(length)]["mean_entropy_by_head"])
        # mat shape: [num_layers, num_heads]
        im = ax.imshow(
            mat,
            aspect="auto",
            origin="upper",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"len = {length}", fontsize=_LABEL_FS)
        ax.set_xlabel("Head index", fontsize=_TICK_FS)
        ax.set_ylabel("Layer index", fontsize=_TICK_FS)
        ax.tick_params(labelsize=_TICK_FS)

    # hide unused axes
    for ax_idx in range(len(lengths), nrows * ncols):
        row, col = divmod(ax_idx, ncols)
        axes[row][col].set_visible(False)

    # shared colour bar
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label("Mean entropy (nats)", fontsize=_LABEL_FS)

    fig.suptitle("Attention entropy per (layer, head)", fontsize=_TITLE_FS, y=1.01)
    _savefig(fig, os.path.join(out_dir, f"fig2_entropy_heatmap_head_layer.{fmt}"), dpi)


# ---------------------------------------------------------------------------
# Figure 3 – entropy vs token position
# ---------------------------------------------------------------------------


def plot_entropy_vs_position(
    data: dict,
    out_dir: str,
    fmt: str,
    dpi: int,
    stride: int = 8,
) -> None:
    """
    Fig 3: x = token position, y = entropy (mean over layers & heads),
           one curve per seq-length.

    *stride* controls sub-sampling of the position axis for readability.

    Interpretation
    ^^^^^^^^^^^^^^
    Very early positions often show low entropy because only a few keys are
    available.  A rising trend followed by a plateau is typical of causal LMs.
    Longer sequences may show entropy growth or saturation at different rates.
    """
    lengths = data["lengths"]
    colours = _colour_cycle(len(lengths))

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    for colour, length in zip(colours, lengths):
        # by_position: [num_layers, seq_len]
        by_pos = np.array(data["results"][str(length)]["entropy_by_position"])
        mean_pos = by_pos.mean(axis=0)  # [seq_len]
        positions = np.arange(len(mean_pos))
        ax.plot(
            positions[::stride],
            mean_pos[::stride],
            label=f"len={length}",
            color=colour,
            linewidth=1.5,
            alpha=0.9,
        )

    ax.set_xlabel("Token position", fontsize=_LABEL_FS)
    ax.set_ylabel("Mean attention entropy (nats)", fontsize=_LABEL_FS)
    ax.set_title(
        "Attention entropy vs token position\n(mean over all layers & heads)",
        fontsize=_TITLE_FS,
    )
    ax.legend(fontsize=_LEGEND_FS, framealpha=0.7)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    _savefig(fig, os.path.join(out_dir, f"fig3_entropy_vs_position.{fmt}"), dpi)


# ---------------------------------------------------------------------------
# Figure 4 – global mean entropy vs sequence length (bar chart)
# ---------------------------------------------------------------------------


def plot_entropy_vs_seqlen(
    data: dict,
    out_dir: str,
    fmt: str,
    dpi: int,
) -> None:
    """
    Fig 4: x = sequence length, y = global mean entropy (scalar per length).

    Interpretation
    ^^^^^^^^^^^^^^
    An increasing trend indicates that longer context forces the model to
    spread attention (more uniform distributions → higher entropy).
    A flat or decreasing trend suggests the model has learned to localise
    attention regardless of context size.
    """
    lengths = data["lengths"]
    global_means = []
    for length in lengths:
        me = np.array(data["results"][str(length)]["mean_entropy_by_layer"])
        global_means.append(float(me.mean()))

    colours = _colour_cycle(len(lengths))
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        [str(l) for l in lengths],
        global_means,
        color=colours,
        edgecolor="white",
        linewidth=0.8,
    )
    # annotate bars with value
    for bar, val in zip(bars, global_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(global_means) * 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=_TICK_FS,
        )

    ax.set_xlabel("Sequence length (tokens)", fontsize=_LABEL_FS)
    ax.set_ylabel("Global mean attention entropy (nats)", fontsize=_LABEL_FS)
    ax.set_title("Global mean entropy vs sequence length", fontsize=_TITLE_FS)
    ax.set_ylim(0, max(global_means) * 1.15)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    _savefig(fig, os.path.join(out_dir, f"fig4_entropy_vs_seqlen.{fmt}"), dpi)


# ---------------------------------------------------------------------------
# Figure 5 – per-layer entropy box plots
# ---------------------------------------------------------------------------


def plot_entropy_boxplot_by_layer(
    data: dict,
    out_dir: str,
    fmt: str,
    dpi: int,
) -> None:
    """
    Fig 5: box plots of entropy distribution across heads × token positions,
           one sub-plot per sequence length.

    Interpretation
    ^^^^^^^^^^^^^^
    Wide boxes / long whiskers → high variance between heads in that layer.
    Narrow boxes clustered near zero → most heads attend very sharply.
    Outlier points above the whiskers are "diffuse" heads worth inspecting.
    """
    lengths = data["lengths"]
    n = len(lengths)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(_FIGSIZE_WIDE[0], _FIGSIZE_WIDE[1]),
        sharey=True,
    )
    if n == 1:
        axes = [axes]

    for ax, length in zip(axes, lengths):
        # quartiles: [num_layers, 5]  → (min, Q1, med, Q3, max)
        q = np.array(
            data["results"][str(length)]["entropy_quartiles_by_layer"]
        )  # [L, 5]
        num_layers = q.shape[0]

        # matplotlib boxplot from pre-computed statistics
        box_stats = []
        for l_idx in range(num_layers):
            box_stats.append(
                {
                    "med": q[l_idx, 2],
                    "q1": q[l_idx, 1],
                    "q3": q[l_idx, 3],
                    "whislo": q[l_idx, 0],
                    "whishi": q[l_idx, 4],
                    "fliers": [],  # already min/max
                    "mean": q[l_idx, 2],  # median as proxy
                    "label": str(l_idx),
                }
            )

        ax.bxp(
            box_stats,
            positions=list(range(num_layers)),
            widths=0.6,
            showfliers=False,
            boxprops=dict(color="steelblue"),
            whiskerprops=dict(color="steelblue", linestyle="--"),
            medianprops=dict(color="tomato", linewidth=2),
            capprops=dict(color="steelblue"),
        )

        # colour-fill boxes by median entropy
        cmap = plt.cm.viridis
        med_vals = q[:, 2]
        norm_med = (med_vals - med_vals.min()) / (
            med_vals.max() - med_vals.min() + 1e-9
        )
        for patch, norm_val in zip(ax.patches, norm_med):
            patch.set_facecolor(cmap(norm_val))
            patch.set_alpha(0.5)

        ax.set_title(f"len = {length}", fontsize=_LABEL_FS)
        ax.set_xlabel("Layer index", fontsize=_TICK_FS)
        if ax is axes[0]:
            ax.set_ylabel("Attention entropy (nats)", fontsize=_TICK_FS)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
        ax.tick_params(labelsize=_TICK_FS)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)

    fig.suptitle(
        "Per-layer entropy distribution (heads × positions)",
        fontsize=_TITLE_FS,
        y=1.02,
    )
    _savefig(fig, os.path.join(out_dir, f"fig5_entropy_boxplot_by_layer.{fmt}"), dpi)


# ---------------------------------------------------------------------------
# Figure 6 – top-k concentration vs token position
# ---------------------------------------------------------------------------


def plot_top_k_concentration_vs_position(
    data: dict,
    out_dir: str,
    fmt: str,
    dpi: int,
    stride: int = 8,
) -> None:
    """
    Fig 6: x = token position, y = fraction of attention mass in top-k keys,
           one curve per sequence length.

    Interpretation
    ^^^^^^^^^^^^^^
    Values near 1.0 at position t → the head attends to very few tokens
    (sharp / local).  Values near k/t → nearly uniform attention.
    This metric complements entropy: both should tell a consistent story.
    """
    lengths = data["lengths"]
    top_k = data.get("top_k", 10)
    colours = _colour_cycle(len(lengths))

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    for colour, length in zip(colours, lengths):
        conc = np.array(
            data["results"][str(length)]["top_k_concentration"]
        )  # [seq_len]
        positions = np.arange(len(conc))
        ax.plot(
            positions[::stride],
            conc[::stride],
            label=f"len={length}",
            color=colour,
            linewidth=1.5,
            alpha=0.9,
        )

    ax.set_xlabel("Token position", fontsize=_LABEL_FS)
    ax.set_ylabel(f"Top-{top_k} attention mass fraction", fontsize=_LABEL_FS)
    ax.set_title(
        f"Attention concentration (top-{top_k} keys) vs token position\n"
        "(mean over all layers & heads)",
        fontsize=_TITLE_FS,
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=_LEGEND_FS, framealpha=0.7)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    _savefig(
        fig,
        os.path.join(out_dir, f"fig6_top{top_k}_concentration_vs_position.{fmt}"),
        dpi,
    )


# ---------------------------------------------------------------------------
# Figure 7 – Δentropy heat-map  (layer × seq-length)
# ---------------------------------------------------------------------------


def plot_delta_entropy_heatmap(
    data: dict,
    out_dir: str,
    fmt: str,
    dpi: int,
) -> None:
    """
    Fig 7: heatmap of Δmean_entropy = entropy(len) − entropy(min_len).

    x-axis = sequence length (excluding baseline = lengths[0])
    y-axis = layer index
    colour = change in mean entropy relative to the shortest context.

    Interpretation
    ^^^^^^^^^^^^^^
    Positive (bright) cells → that layer's attention became *more diffuse*
    when context grows.  Negative (dark) cells → attention sharpened.
    Layers with large positive Δentropy are most sensitive to context length
    and are the primary targets for RoPE-extension improvements.
    """
    lengths = data["lengths"]
    baseline = str(lengths[0])
    base_arr = np.array(
        data["results"][baseline]["mean_entropy_by_layer"]
    )  # [num_layers]

    longer = lengths[1:]  # exclude baseline
    delta_mat = np.zeros((len(base_arr), len(longer)))  # [num_layers, n_longer]

    for col, length in enumerate(longer):
        me = np.array(data["results"][str(length)]["mean_entropy_by_layer"])
        delta_mat[:, col] = me - base_arr

    fig, ax = plt.subplots(figsize=(max(6, len(longer) * 2.5 + 2), 7))
    im = ax.imshow(
        delta_mat,
        aspect="auto",
        origin="upper",
        cmap="RdBu_r",
        vmin=-np.abs(delta_mat).max(),
        vmax=np.abs(delta_mat).max(),
    )

    ax.set_xticks(range(len(longer)))
    ax.set_xticklabels([str(l) for l in longer], fontsize=_TICK_FS)
    ax.set_yticks(range(0, len(base_arr), max(1, len(base_arr) // 8)))
    ax.set_yticklabels(
        [str(i) for i in range(0, len(base_arr), max(1, len(base_arr) // 8))],
        fontsize=_TICK_FS,
    )
    ax.set_xlabel("Sequence length (tokens)", fontsize=_LABEL_FS)
    ax.set_ylabel("Layer index", fontsize=_LABEL_FS)
    ax.set_title(
        f"Δ Mean entropy relative to len={lengths[0]}\n"
        "(red = more diffuse, blue = sharper)",
        fontsize=_TITLE_FS,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Δ entropy (nats)", fontsize=_LABEL_FS)

    _savefig(fig, os.path.join(out_dir, f"fig7_delta_entropy_heatmap.{fmt}"), dpi)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

import math  # needed by plot_entropy_heatmap_head_layer


_PLOT_FUNCTIONS = [
    plot_mean_entropy_by_layer,
    plot_entropy_heatmap_head_layer,
    plot_entropy_vs_position,
    plot_entropy_vs_seqlen,
    plot_entropy_boxplot_by_layer,
    plot_top_k_concentration_vs_position,
    plot_delta_entropy_heatmap,
]


def plot_all(data: dict, out_dir: str, fmt: str = "png", dpi: int = 120) -> None:
    """
    Generate all seven figures for the given entropy JSON data.

    Parameters
    ----------
    data    : dict  loaded from the JSON produced by EntropyEvaluator.evaluate()
    out_dir : str   directory for output images
    fmt     : str   image format ('png', 'pdf', 'svg')
    dpi     : int   resolution (ignored for vector formats)
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nGenerating {len(_PLOT_FUNCTIONS)} figures → {out_dir}/")

    # Fig 7 requires at least 2 lengths
    funcs = _PLOT_FUNCTIONS
    if len(data["lengths"]) < 2:
        print("  [warn] Only 1 length in data; skipping Fig 7 (delta heatmap).")
        funcs = _PLOT_FUNCTIONS[:-1]

    for fn in funcs:
        try:
            fn(data, out_dir, fmt, dpi)
        except Exception as exc:
            print(f"  [error] {fn.__name__}: {exc}")

    print("Done.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot attention-entropy results from EntropyEvaluator JSON output."
    )
    p.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        default="results/entropy/llama-7b_none.json",
        help="Path to JSON file produced by eval/entropy.py.",
    )
    p.add_argument(
        "--out-dir",
        "-o",
        type=str,
        default="results/entropy/plots",
        help=(
            "Output directory for figures.  "
            "Defaults to <input_dir>/plots/<input_stem>."
        ),
    )
    p.add_argument(
        "--fmt",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output image format (default: png).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Resolution for raster formats (default: 120).",
    )
    p.add_argument(
        "--fig",
        type=int,
        default=None,
        choices=list(range(1, 8)),
        help="Generate only figure N (1–7); omit to generate all.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r") as f:
        data = json.load(f)

    out_dir = args.out_dir or str(input_path.parent / "plots" / input_path.stem)

    if args.fig is not None:
        fn = _PLOT_FUNCTIONS[args.fig - 1]
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nGenerating figure {args.fig} → {out_dir}/")
        fn(data, out_dir, args.fmt, args.dpi)
    else:
        plot_all(data, out_dir, fmt=args.fmt, dpi=args.dpi)


if __name__ == "__main__":
    main()
