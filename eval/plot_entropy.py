"""
eval/plot_entropy.py
--------------------
Generate seven diagnostic figures from a JSON file produced by EntropyEvaluator.

Usage
-----
    # all 7 figures
    python eval/plot_entropy.py --input results/entropy/llama-7b_none.json

    # single figure
    python eval/plot_entropy.py --input results/entropy/llama-7b_none.json --fig 3

    # custom output dir and format
    python eval/plot_entropy.py --input results/entropy/llama-7b_none.json \\
        --out-dir figures/entropy --fmt pdf --dpi 250

Seven figures produced
----------------------
Fig 1  fig01_layer_depth_curve_{raw|norm}.png
       Line plot – x=layer, y=mean entropy, one curve per seq-length.
       Raw and normalised versions side-by-side in a single figure (1×2).

Fig 2  fig02_head_layer_heatmap_{raw|norm}.png
       Heat-map grid – x=head, y=layer, colour=mean entropy.
       Separate figures for raw and normalised; each has 2×2 sub-plots (4 lengths).

Fig 3  fig03_entropy_vs_position_{raw|norm}.png
       Line plot – x=token-position, y=entropy, one curve per seq-length.
       Auto-selects the 4 most-specialised layers (highest head std) as 2×2 sub-plots.
       Gray band marks the early-position region where t < top_k.

Fig 4  fig04_head_norm_std_by_layer.png
       Line plot – x=layer, y=head-entropy std (normalised), one curve per seq-length.
       Single figure; shows which layers have the highest intra-layer head diversity.

Fig 5  fig05_position_head_heatmap_{raw|norm}.png
       Heat-map – x=token-position, y=head, colour=entropy.
       Selects the single highest-std layer; 2×2 sub-plots for the 4 seq-lengths.

Fig 6  fig06_delta_entropy_heatmap_{raw|norm}.png
       Delta heat-map – x=seq-length (vs baseline=shortest), y=layer.
       Red = more diffuse, blue = sharper relative to baseline.
       Separate figures for raw and normalised.

Fig 7  fig07_entropy_boxplot_violin_norm.png
       Box + violin hybrid – x=layer, y=normalised entropy distribution.
       2×2 sub-plots (4 seq-lengths).  Violin shows full distribution shape.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import gaussian_kde  # for smooth violin approximation

# ---------------------------------------------------------------------------
# Global aesthetics
# ---------------------------------------------------------------------------

# Fixed colours for the four experiment sequence lengths
_LENGTH_COLORS: dict[int, str] = {
    256: "#1565C0",  # dark blue
    512: "#E65100",  # deep orange
    1024: "#2E7D32",  # dark green
    2048: "#B71C1C",  # dark red
    3072: "#6A1B9A",  # deep purple
}
_FALLBACK_CMAP = plt.cm.tab10  # for unexpected lengths

# Figure sizes  (width, height) in inches – intentionally large for clarity
_FS_12 = (32, 14)  # 1×2 layout
_FS_22 = (30, 26)  # 2×2 layout
_FS_22SM = (28, 22)  # 2×2 for violin/box (slightly smaller per-panel)
_FS_HEAT = (32, 26)  # 2×2 heatmap
_FS_SINGLE = (22, 13)  # single panel
_FS_DELTA = (18, 22)  # delta heatmap (tall)

_DPI = 300
_TITLE_FS = 20
_SUPTITLE_FS = 24
_LABEL_FS = 17
_TICK_FS = 13
_LEGEND_FS = 14
_ANNOT_FS = 12
_ENDLABEL_FS = 12

# Number of violin sample points for KDE rendering
_VIOLIN_POINTS = 300


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _get_color(seq_len: int, lengths: Sequence[int]) -> str:
    """Return a consistent colour for a given sequence length."""
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
    fs: int = _ENDLABEL_FS,
) -> None:
    """Annotate the last point of a line with its label."""
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


def _shade_boundary(ax: plt.Axes, boundary: int, alpha: float = 0.10) -> None:
    """
    Draw a light gray rectangle over x ∈ [0, boundary] to indicate the
    early-position region where top-k concentration is forced to 1.0.
    """
    if boundary <= 0:
        return
    ax.axvspan(
        0,
        boundary,
        color="gray",
        alpha=alpha,
        zorder=0,
        label=f"forced conc. region (t<{boundary+1})",
    )


def _norm_from_raw_np(arr: np.ndarray) -> np.ndarray:
    """
    Compute H_norm = H / log(t+1) along the last axis of a numpy array.

    Parameters
    ----------
    arr : ndarray  shape [..., T]

    Returns
    -------
    ndarray  shape [..., T]  values in [0, 1]
    """
    T = arr.shape[-1]
    pos = np.arange(T, dtype=np.float64)
    den = np.log(pos + 1.0).clip(min=1e-6)
    return arr / den


def _auto_select_layers(data: dict, n: int = 4) -> List[int]:
    """
    Return the indices of the *n* layers with the highest average
    head-entropy standard deviation across all evaluated sequence lengths.

    These layers show the most intra-layer head specialisation and are the
    most informative to inspect in detail.
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


def _grid(n: int):
    """Return (nrows, ncols) for n sub-plots with preferred 2×2 / 3×3 layout."""
    if n <= 1:
        return 1, 1
    if n <= 2:
        return 1, 2
    if n <= 4:
        return 2, 2
    if n <= 6:
        return 2, 3
    if n <= 9:
        return 3, 3
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def _common_vrange(matrices: List[np.ndarray]):
    """Compute a shared (vmin, vmax) across a list of 2-D arrays."""
    all_vals = np.concatenate([m.ravel() for m in matrices])
    return float(np.nanmin(all_vals)), float(np.nanmax(all_vals))


# ---------------------------------------------------------------------------
# Figure 1 – Layer depth entropy curve
# ---------------------------------------------------------------------------


def plot_layer_depth_curve(data: dict, out_dir: str, fmt: str, dpi: int) -> None:
    """
    Fig 1 — x=layer index, y=mean entropy, one line per seq-length.

    Layout: 1×2 (raw left, normalised right).
    Each curve represents the mean over all token positions and all heads,
    showing the per-layer entropy profile of the network.

    Interpretation
    ^^^^^^^^^^^^^^
    A U-shape or monotone decrease indicates depth-wise sharpening.
    Vertical separation between coloured curves at the same layer shows
    how sensitive that layer is to context length.
    """
    lengths = data["lengths"]
    L = data["num_layers"]
    layer_x = np.arange(L)

    fig, axes = plt.subplots(1, 2, figsize=_FS_12)

    for ax_idx, (key, ylabel, title) in enumerate(
        [
            (
                "entropy_layer_position",
                "Mean entropy (nats)",
                "Raw Shannon entropy  H(l)",
            ),
            (
                "norm_entropy_layer_position",
                "Mean normalised entropy  H_norm ∈ [0,1]",
                "Normalised entropy  H_norm(l)",
            ),
        ]
    ):
        ax = axes[ax_idx]
        for sl in lengths:
            col = _get_color(sl, lengths)
            matrix = np.array(data["results"][str(sl)][key])  # [L, T]
            curve = matrix.mean(axis=1)  # [L]
            ax.plot(
                layer_x,
                curve,
                color=col,
                linewidth=2.2,
                marker="o",
                markersize=5,
                label=f"len={sl}",
            )
            _end_label(ax, layer_x, curve, str(sl), col)

        ax.set_xlabel("Layer index", fontsize=_LABEL_FS)
        ax.set_ylabel(ylabel, fontsize=_LABEL_FS)
        ax.set_title(title, fontsize=_TITLE_FS, pad=10)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
        ax.tick_params(labelsize=_TICK_FS)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.55)
        ax.legend(fontsize=_LEGEND_FS, framealpha=0.8)

    fig.suptitle(
        "Fig 1 — Attention entropy profile across network depth",
        fontsize=_SUPTITLE_FS,
        y=1.01,
    )
    _savefig(fig, os.path.join(out_dir, f"fig01_layer_depth_curve.{fmt}"), dpi)


# ---------------------------------------------------------------------------
# Figure 2 – Head × Layer heatmap
# ---------------------------------------------------------------------------


def _plot_head_layer_heatmap_one(
    data: dict,
    key: str,
    title_prefix: str,
    ylabel: str,
    out_dir: str,
    fname: str,
    fmt: str,
    dpi: int,
    cmap: str = "viridis",
    vrange_fixed: tuple | None = None,
) -> None:
    """Internal worker for Fig 2 (raw and norm share the same layout)."""
    lengths = data["lengths"]
    nrows, ncols = _grid(len(lengths))

    mats = [np.array(data["results"][str(sl)][key]) for sl in lengths]
    vmin, vmax = vrange_fixed or _common_vrange(mats)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=_FS_HEAT,
        squeeze=False,
        constrained_layout=True,
    )

    for i, (sl, mat) in enumerate(zip(lengths, mats)):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        im = ax.imshow(
            mat,
            aspect="auto",
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"seq_len = {sl}", fontsize=_LABEL_FS, pad=6)
        ax.set_xlabel("Head index", fontsize=_TICK_FS)
        ax.set_ylabel("Layer index", fontsize=_TICK_FS)
        ax.tick_params(labelsize=_TICK_FS)

    # hide unused cells
    for i in range(len(lengths), nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r][c].set_visible(False)

    cbar = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.06, shrink=0.82, aspect=40)
    cbar.set_label(ylabel, fontsize=_LABEL_FS)
    cbar.ax.tick_params(labelsize=_TICK_FS)

    fig.suptitle(
        f"Fig 2 — {title_prefix}: head × layer heatmap\n"
        "Each cell = mean entropy of that (layer, head) across all positions and samples",
        fontsize=_SUPTITLE_FS,
        y=1.02,
    )
    _savefig(fig, os.path.join(out_dir, fname), dpi, constrained_layout=True)


def plot_head_layer_heatmap(data: dict, out_dir: str, fmt: str, dpi: int) -> None:
    """Fig 2 — two files: raw entropy and normalised entropy."""
    _plot_head_layer_heatmap_one(
        data,
        "entropy_head_layer",
        title_prefix="Raw entropy",
        ylabel="Mean H (nats)",
        out_dir=out_dir,
        fname=f"fig02_head_layer_heatmap_raw.{fmt}",
        fmt=fmt,
        dpi=dpi,
    )
    _plot_head_layer_heatmap_one(
        data,
        "norm_entropy_head_layer",
        title_prefix="Normalised entropy",
        ylabel="Mean H_norm ∈ [0,1]",
        out_dir=out_dir,
        fname=f"fig02_head_layer_heatmap_norm.{fmt}",
        fmt=fmt,
        dpi=dpi,
        vrange_fixed=(0.0, 1.0),
    )


# ---------------------------------------------------------------------------
# Figure 3 – Entropy vs token position  (selected layers)
# ---------------------------------------------------------------------------


def _plot_entropy_vs_position_one(
    data: dict,
    key: str,
    ylabel: str,
    title_prefix: str,
    selected_layers: List[int],
    out_dir: str,
    fname: str,
    fmt: str,
    dpi: int,
) -> None:
    """Internal worker for Fig 3."""
    lengths = data["lengths"]
    nrows, ncols = _grid(len(selected_layers))
    top_k_boundary = max(data["results"][str(sl)]["top_k_boundary"] for sl in lengths)

    fig, axes = plt.subplots(nrows, ncols, figsize=_FS_22, squeeze=False)

    for ax_i, layer_idx in enumerate(selected_layers):
        r, c = divmod(ax_i, ncols)
        ax = axes[r][c]

        _shade_boundary(ax, top_k_boundary)

        for sl in lengths:
            col = _get_color(sl, lengths)
            matrix = np.array(data["results"][str(sl)][key])  # [L, T]
            T = matrix.shape[1]
            curve = matrix[layer_idx]  # [T]
            pos_x = np.arange(T)
            ax.plot(
                pos_x,
                curve,
                color=col,
                linewidth=1.8,
                alpha=0.9,
                label=f"len={sl}",
            )
            _end_label(ax, pos_x, curve, str(sl), col)

        ax.set_title(f"Layer {layer_idx}", fontsize=_LABEL_FS, pad=6)
        ax.set_xlabel("Token position", fontsize=_TICK_FS)
        ax.set_ylabel(ylabel, fontsize=_TICK_FS)
        ax.tick_params(labelsize=_TICK_FS)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=_LEGEND_FS - 1, framealpha=0.8)

    for ax_i in range(len(selected_layers), nrows * ncols):
        r, c = divmod(ax_i, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(
        f"Fig 3 — {title_prefix}: entropy vs token position\n"
        "Layers selected by highest head-entropy std (most specialised).\n"
        "Gray band = forced top-k region (t < top_k)",
        fontsize=_SUPTITLE_FS,
        y=1.02,
    )
    _savefig(fig, os.path.join(out_dir, fname), dpi)


def plot_entropy_vs_position(
    data: dict,
    out_dir: str,
    fmt: str,
    dpi: int,
    selected_layers: List[int] | None = None,
) -> None:
    """Fig 3 — two files: raw and normalised."""
    layers = selected_layers or _auto_select_layers(data, n=4)
    print(f"  Fig 3 auto-selected layers: {layers}")

    _plot_entropy_vs_position_one(
        data,
        "entropy_layer_position",
        ylabel="H (nats)",
        title_prefix="Raw entropy",
        selected_layers=layers,
        out_dir=out_dir,
        fname=f"fig03_entropy_vs_position_raw.{fmt}",
        fmt=fmt,
        dpi=dpi,
    )
    _plot_entropy_vs_position_one(
        data,
        "norm_entropy_layer_position",
        ylabel="H_norm ∈ [0,1]",
        title_prefix="Normalised entropy",
        selected_layers=layers,
        out_dir=out_dir,
        fname=f"fig03_entropy_vs_position_norm.{fmt}",
        fmt=fmt,
        dpi=dpi,
    )


# ---------------------------------------------------------------------------
# Figure 4 – Head-entropy std by layer
# ---------------------------------------------------------------------------


def plot_head_norm_std_by_layer(
    data: dict,
    out_dir: str,
    fmt: str,
    dpi: int,
) -> None:
    """
    Fig 4 — x=layer, y=std of per-head normalised entropy across heads,
             one line per seq-length.

    A high std at layer l means the heads in that layer have very different
    attention patterns (strong specialisation).  A low std means they behave
    similarly (possible redundancy).
    """
    lengths = data["lengths"]
    L = data["num_layers"]
    layer_x = np.arange(L)

    fig, ax = plt.subplots(figsize=_FS_SINGLE)

    for sl in lengths:
        col = _get_color(sl, lengths)
        std = np.array(data["results"][str(sl)]["head_norm_std_by_layer"])
        ax.plot(
            layer_x,
            std,
            color=col,
            linewidth=2.2,
            marker="s",
            markersize=5,
            label=f"len={sl}",
        )
        _end_label(ax, layer_x, std, str(sl), col)

    ax.set_xlabel("Layer index", fontsize=_LABEL_FS)
    ax.set_ylabel("Std of per-head  H_norm  (across heads)", fontsize=_LABEL_FS)
    ax.set_title(
        "Fig 4 — Intra-layer head specialisation\n"
        "Higher std = heads in this layer have more diverse attention patterns",
        fontsize=_TITLE_FS,
        pad=10,
    )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax.tick_params(labelsize=_TICK_FS)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.55)
    ax.legend(fontsize=_LEGEND_FS, framealpha=0.8)

    _savefig(fig, os.path.join(out_dir, f"fig04_head_norm_std_by_layer.{fmt}"), dpi)


# ---------------------------------------------------------------------------
# Figure 5 – Position × Head heatmap  (top-variance layer)
# ---------------------------------------------------------------------------


def _plot_position_head_heatmap_one(
    data: dict,
    top_layer: int,
    ylabel: str,
    title_prefix: str,
    is_norm: bool,
    out_dir: str,
    fname: str,
    fmt: str,
    dpi: int,
) -> None:
    """Internal worker for Fig 5."""
    lengths = data["lengths"]
    nrows, ncols = _grid(len(lengths))
    top_k_bnd = max(data["results"][str(sl)]["top_k_boundary"] for sl in lengths)

    # collect all matrices for shared colour range
    all_mats = []
    for sl in lengths:
        raw = np.array(data["results"][str(sl)]["entropy_head_layer_position"])[
            top_layer
        ]  # [H, T]
        mat = _norm_from_raw_np(raw) if is_norm else raw
        all_mats.append(mat)

    vmin, vmax = _common_vrange(all_mats)
    if is_norm:
        vmin, vmax = 0.0, 1.0  # fixed scale for norm

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=_FS_HEAT,
        squeeze=False,
        constrained_layout=True,
    )

    for i, (sl, mat) in enumerate(zip(lengths, all_mats)):
        r, c = divmod(i, ncols)
        ax = axes[r][c]

        im = ax.imshow(
            mat,
            aspect="auto",
            origin="upper",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        # vertical line at top_k boundary
        if top_k_bnd > 0:
            ax.axvline(
                top_k_bnd,
                color="white",
                linestyle="--",
                linewidth=1.2,
                alpha=0.7,
            )
        ax.set_title(f"seq_len = {sl}", fontsize=_LABEL_FS, pad=6)
        ax.set_xlabel("Token position", fontsize=_TICK_FS)
        ax.set_ylabel("Head index", fontsize=_TICK_FS)
        ax.tick_params(labelsize=_TICK_FS)

    for i in range(len(lengths), nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r][c].set_visible(False)

    cbar = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.06, shrink=0.82, aspect=40)
    cbar.set_label(ylabel, fontsize=_LABEL_FS)
    cbar.ax.tick_params(labelsize=_TICK_FS)

    fig.suptitle(
        f"Fig 5 — {title_prefix}: position × head heatmap  (layer {top_layer})\n"
        "Highest-std layer selected automatically.  "
        "White dashed line = top-k boundary.",
        fontsize=_SUPTITLE_FS,
        y=1.02,
    )
    _savefig(fig, os.path.join(out_dir, fname), dpi, constrained_layout=True)


def plot_position_head_heatmap(
    data: dict,
    out_dir: str,
    fmt: str,
    dpi: int,
    selected_layers: List[int] | None = None,
) -> None:
    """
    Fig 5 — two files: raw and normalised.

    Selects the single layer with the highest intra-layer head std
    (most informative layer to inspect at token-position resolution).
    """
    top_layer = (selected_layers or _auto_select_layers(data, n=1))[0]
    print(f"  Fig 5 top-variance layer: {top_layer}")

    _plot_position_head_heatmap_one(
        data,
        top_layer,
        ylabel="H (nats)",
        title_prefix="Raw entropy",
        is_norm=False,
        out_dir=out_dir,
        fname=f"fig05_position_head_heatmap_raw.{fmt}",
        fmt=fmt,
        dpi=dpi,
    )
    _plot_position_head_heatmap_one(
        data,
        top_layer,
        ylabel="H_norm ∈ [0,1]",
        title_prefix="Normalised entropy",
        is_norm=True,
        out_dir=out_dir,
        fname=f"fig05_position_head_heatmap_norm.{fmt}",
        fmt=fmt,
        dpi=dpi,
    )


# ---------------------------------------------------------------------------
# Figure 6 – Delta entropy heatmap
# ---------------------------------------------------------------------------


def _plot_delta_heatmap_one(
    data: dict,
    key: str,
    ylabel: str,
    title_prefix: str,
    out_dir: str,
    fname: str,
    fmt: str,
    dpi: int,
) -> None:
    """Internal worker for Fig 6."""
    lengths = data["lengths"]
    baseline = str(lengths[0])
    longer = lengths[1:]
    L = data["num_layers"]

    # baseline layer means
    base_mat = np.array(data["results"][baseline][key])  # [L, T] or [L, H]
    base_mean = base_mat.mean(axis=1)  # [L]

    delta = np.zeros((L, len(longer)))
    for col_i, sl in enumerate(longer):
        mat = np.array(data["results"][str(sl)][key])
        delta[:, col_i] = mat.mean(axis=1) - base_mean

    abs_max = max(np.abs(delta).max(), 1e-8)

    fig, ax = plt.subplots(figsize=_FS_DELTA)
    im = ax.imshow(
        delta,
        aspect="auto",
        origin="upper",
        cmap="RdBu_r",
        vmin=-abs_max,
        vmax=abs_max,
    )

    ax.set_xticks(range(len(longer)))
    ax.set_xticklabels([str(sl) for sl in longer], fontsize=_TICK_FS)
    ax.set_xlabel("Sequence length (tokens)", fontsize=_LABEL_FS)

    y_step = max(1, L // 8)
    ax.set_yticks(range(0, L, y_step))
    ax.set_yticklabels([str(i) for i in range(0, L, y_step)], fontsize=_TICK_FS)
    ax.set_ylabel("Layer index", fontsize=_LABEL_FS)

    ax.set_title(
        f"Fig 6 — {title_prefix}: Δ entropy vs baseline (seq_len={lengths[0]})\n"
        f"Red = more diffuse  |  Blue = sharper  |  White = no change",
        fontsize=_TITLE_FS,
        pad=10,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label(f"Δ {ylabel}", fontsize=_LABEL_FS)
    cbar.ax.tick_params(labelsize=_TICK_FS)

    # annotate cells with Δ value
    for row in range(L):
        for col_i in range(len(longer)):
            v = delta[row, col_i]
            ax.text(
                col_i,
                row,
                f"{v:+.3f}",
                ha="center",
                va="center",
                fontsize=max(6, _ANNOT_FS - 2 - L // 16),
                color="white" if abs(v) > abs_max * 0.4 else "black",
            )

    _savefig(fig, os.path.join(out_dir, fname), dpi)


def plot_delta_entropy_heatmap(
    data: dict,
    out_dir: str,
    fmt: str,
    dpi: int,
) -> None:
    """Fig 6 — two files: raw and normalised."""
    if len(data["lengths"]) < 2:
        print("  Fig 6: skipped (need at least 2 lengths).")
        return

    _plot_delta_heatmap_one(
        data,
        "entropy_layer_position",
        ylabel="H (nats)",
        title_prefix="Raw entropy",
        out_dir=out_dir,
        fname=f"fig06_delta_entropy_heatmap_raw.{fmt}",
        fmt=fmt,
        dpi=dpi,
    )
    _plot_delta_heatmap_one(
        data,
        "norm_entropy_layer_position",
        ylabel="H_norm",
        title_prefix="Normalised entropy",
        out_dir=out_dir,
        fname=f"fig06_delta_entropy_heatmap_norm.{fmt}",
        fmt=fmt,
        dpi=dpi,
    )


# ---------------------------------------------------------------------------
# Figure 7 – Box + Violin  (normalised entropy distribution)
# ---------------------------------------------------------------------------


def _violin_from_data(ax, positions, datasets, color, width=0.7, alpha=0.45):
    """
    Draw a smooth violin (via scipy KDE) behind a boxplot for a set of
    (position, data) pairs.

    Parameters
    ----------
    ax        : matplotlib Axes
    positions : list[int]   x positions for each violin
    datasets  : list[ndarray]  1-D data for each position
    color     : str
    width     : float   half-width of each violin
    alpha     : float   transparency
    """
    for pos, vals in zip(positions, datasets):
        if len(vals) < 4:
            continue
        try:
            kde = gaussian_kde(vals, bw_method="scott")
            y_lo, y_hi = vals.min(), vals.max()
            ys = np.linspace(y_lo, y_hi, _VIOLIN_POINTS)
            dens = kde(ys)
            # scale density to desired visual width
            max_d = dens.max()
            if max_d > 0:
                dens = dens / max_d * width
            ax.fill_betweenx(
                ys,
                pos - dens,
                pos + dens,
                color=color,
                alpha=alpha,
                linewidth=0,
            )
        except Exception:
            pass  # skip gracefully if KDE fails (e.g. constant data)


def _plot_boxplot_violin_one(
    data: dict,
    lengths: list,
    seq_len: int,
    ax: plt.Axes,
) -> None:
    """
    Draw box + violin for one seq_len subplot.

    Data: for each layer, extract H×T normalised entropy values from
    entropy_head_layer_position (computing norm on-the-fly).
    """
    col = _get_color(seq_len, lengths)
    raw_3d = np.array(
        data["results"][str(seq_len)]["entropy_head_layer_position"]
    )  # [L, H, T]
    norm_3d = _norm_from_raw_np(raw_3d)  # [L, H, T]
    L = norm_3d.shape[0]
    layer_pos = list(range(L))

    # per-layer 1-D arrays for violin
    layer_datasets = [norm_3d[l].ravel() for l in range(L)]

    _violin_from_data(ax, layer_pos, layer_datasets, col, width=0.42, alpha=0.40)

    # box from pre-computed quartiles
    q_arr = np.array(
        data["results"][str(seq_len)]["norm_entropy_quartiles_by_layer"]
    )  # [L, 5]
    box_stats = []
    for l in range(L):
        box_stats.append(
            {
                "med": q_arr[l, 2],
                "q1": q_arr[l, 1],
                "q3": q_arr[l, 3],
                "whislo": q_arr[l, 0],
                "whishi": q_arr[l, 4],
                "fliers": [],
                "mean": q_arr[l, 2],
                "label": str(l),
            }
        )
    ax.bxp(
        box_stats,
        positions=layer_pos,
        widths=0.28,
        showfliers=False,
        boxprops=dict(color=col, linewidth=1.8),
        whiskerprops=dict(color=col, linewidth=1.5, linestyle="--"),
        medianprops=dict(color="white", linewidth=2.5),
        capprops=dict(color=col, linewidth=1.8),
        patch_artist=True,
    )
    # re-colour box faces
    for patch in ax.patches:
        patch.set_facecolor(col)
        patch.set_alpha(0.65)

    ax.set_title(f"seq_len = {seq_len}", fontsize=_LABEL_FS, pad=6)
    ax.set_xlabel("Layer index", fontsize=_TICK_FS)
    ax.set_ylabel("H_norm ∈ [0,1]", fontsize=_TICK_FS)
    ax.set_xlim(-0.7, L - 0.3)
    ax.set_ylim(-0.02, 1.08)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax.tick_params(labelsize=_TICK_FS)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)


def plot_entropy_boxplot_violin(
    data: dict,
    out_dir: str,
    fmt: str,
    dpi: int,
) -> None:
    """
    Fig 7 — normalised entropy distribution (box + violin) per layer.
    2×2 sub-plots, one per seq-length.

    The violin shows the full KDE of H_norm values across all heads and
    token positions in the average sample.  The overlaid box shows the
    pre-computed quartiles (min, Q1, median, Q3, max).

    Wide violins at a layer → broad diversity of attention patterns.
    Bimodal violin → coexisting sharp heads and diffuse heads in that layer.
    """
    lengths = data["lengths"]
    nrows, ncols = _grid(len(lengths))

    fig, axes = plt.subplots(nrows, ncols, figsize=_FS_22SM, squeeze=False, sharey=True)

    for i, sl in enumerate(lengths):
        r, c = divmod(i, ncols)
        _plot_boxplot_violin_one(data, lengths, sl, axes[r][c])

    for i in range(len(lengths), nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(
        "Fig 7 — Normalised entropy distribution per layer  (box + violin)\n"
        "Violin = KDE of H_norm across all heads & token positions; "
        "Box = quartiles; white line = median",
        fontsize=_SUPTITLE_FS,
        y=1.02,
    )
    _savefig(
        fig,
        os.path.join(out_dir, f"fig07_entropy_boxplot_violin_norm.{fmt}"),
        dpi,
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_FIG_FUNCTIONS = [
    plot_layer_depth_curve,  # 1
    plot_head_layer_heatmap,  # 2
    plot_entropy_vs_position,  # 3
    plot_head_norm_std_by_layer,  # 4
    plot_position_head_heatmap,  # 5
    plot_delta_entropy_heatmap,  # 6
    plot_entropy_boxplot_violin,  # 7
]


def plot_all(data: dict, out_dir: str, fmt: str = "png", dpi: int = _DPI) -> None:
    """
    Generate all seven figures.

    Parameters
    ----------
    data    : dict   loaded from EntropyEvaluator JSON output
    out_dir : str    destination directory (created if absent)
    fmt     : str    'png' | 'pdf' | 'svg'
    dpi     : int    resolution (raster formats)
    """
    os.makedirs(out_dir, exist_ok=True)

    # pre-compute shared layer selection so Fig 3 and Fig 5 use the same layers
    selected = _auto_select_layers(data, n=4)
    print(f"\nGenerating {len(_FIG_FUNCTIONS)} figures → {out_dir}/")
    print(f"Auto-selected layers for Figs 3 & 5: {selected}")

    for fig_num, fn in enumerate(_FIG_FUNCTIONS, start=1):
        print(f"\n[Fig {fig_num}] {fn.__name__}")
        try:
            if fn is plot_entropy_vs_position:
                fn(data, out_dir, fmt, dpi, selected_layers=selected)
            elif fn is plot_position_head_heatmap:
                fn(data, out_dir, fmt, dpi, selected_layers=selected)
            else:
                fn(data, out_dir, fmt, dpi)
        except Exception as exc:
            print(f"  [ERROR] {exc}")

    print("\nAll figures done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot attention-entropy results from EntropyEvaluator JSON output."
    )
    p.add_argument(
        "--input",
        "-i",
        type=str,
        default="results/entropy/llama-7b_none.json",
        help="Path to JSON produced by eval/entropy.py.",
    )
    p.add_argument(
        "--out-dir",
        "-o",
        type=str,
        default="results/entropy/plots",
        help="Output directory.  Default: <input_dir>/plots/<input_stem>.",
    )
    p.add_argument(
        "--fmt",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Image format (default: png).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=_DPI,
        help=f"Resolution for raster formats (default: {_DPI}).",
    )
    p.add_argument(
        "--fig",
        type=int,
        default=None,
        choices=range(1, 8),
        help="Generate only figure N (1–7); omit for all.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    print(f"Loading {input_path} …")
    with open(input_path, "r") as f:
        data = json.load(f)
    print(
        f"  lengths={data['lengths']}  "
        f"layers={data['num_layers']}  heads={data['num_heads']}"
    )

    out_dir = args.out_dir or str(input_path.parent / "plots" / input_path.stem)

    if args.fig is not None:
        fn = _FIG_FUNCTIONS[args.fig - 1]
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nGenerating Fig {args.fig} → {out_dir}/")
        selected = _auto_select_layers(data, n=4)
        try:
            if fn is plot_entropy_vs_position:
                fn(data, out_dir, args.fmt, args.dpi, selected_layers=selected)
            elif fn is plot_position_head_heatmap:
                fn(data, out_dir, args.fmt, args.dpi, selected_layers=selected)
            else:
                fn(data, out_dir, args.fmt, args.dpi)
        except Exception as exc:
            print(f"[ERROR] {exc}")
    else:
        plot_all(data, out_dir, fmt=args.fmt, dpi=args.dpi)


if __name__ == "__main__":
    main()
