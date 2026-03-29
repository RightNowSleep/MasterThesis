from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

matplotlib.use("Agg")

# ============================================================================ #
#  Global style constants                                                      #
# ============================================================================ #

# --- Typography (matches most IEEE / ACL / NeurIPS templates) ---
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Palatino"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "legend.title_fontsize": 10,
        "figure.dpi": 150,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.45,
        "lines.linewidth": 1.8,
        "lines.markersize": 5.5,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#cccccc",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

# --- Color palette: 16 perceptually distinct, print-safe colors ---
# Ordered by visual contrast; first entries reserved for "canonical" methods
_PALETTE: List[str] = [
    "#2166AC",  # 0  deep blue       → Standard RoPE (baseline)
    "#D6604D",  # 1  brick red       → Linear (PI)
    "#4DAF4A",  # 2  forest green    → NTK-aware
    "#984EA3",  # 3  purple          → NTK-by-parts
    "#FF7F00",  # 4  orange          → YaRN
    "#A65628",  # 5  brown           → My-RoPE
    "#F781BF",  # 6  pink            → My-RoPE (scaled)
    "#999999",  # 7  grey            → My-RoPE2
    "#66C2A5",  # 8  mint            → My-RoPE2 (scaled)
    "#FC8D62",  # 9  salmon          → Block-Layered
    "#8DA0CB",  # 10 steel blue      → Block-Layered (scaled)
    "#E78AC3",  # 11 mauve           → Freq-Smooth
    "#A6D854",  # 12 lime            → Freq-Smooth (scaled)
    "#E41A1C",  # 13 bright red      → Freq-Reciprocal (changed from dark green)
    "#762A83",  # 14 dark purple     → Freq-Reciprocal (scaled)
    "#4393C3",  # 15 sky blue        → Freq-Reciprocal-NoLayer
]

_MARKERS: List[str] = [
    "o",
    "s",
    "^",
    "D",
    "v",
    "P",
    "X",
    "*",
    "h",
    "p",
    "<",
    ">",
    "8",
    "H",
    "d",
    "+",
]

# --- Canonical ordering for legend: none first, then grouped by family ---
_ROPE_DISPLAY_ORDER: List[str] = [
    "none",
    "linear",
    "ntk",
    "part-ntk",
    "yarn",
    "my-rope",
    "my-rope-scaled",
    "my-rope2",
    "my-rope2-scaled",
    "block-layered",
    "block-layered-scaled",
    "freq-smooth",
    "freq-smooth-scaled",
    "freq-reciprocal",
    "freq-reciprocal-scaled",
    "freq-reciprocal-scaled-no-layer",
    "freq-reciprocal-scaled-adaptive",
]

# Map rope_type → (palette_index, display_name)
# Display names follow the paper convention:
#   none              → "RoPE"               (vanilla baseline)
#   linear            → "Linear"             (Position Interpolation)
#   ntk               → "NTK"               (NTK-aware scaling)
#   part-ntk          → "NTK-by-Parts"       (NTK-by-parts)
#   yarn              → "YaRN"
#   freq-reciprocal   → "Freq-Reciprocal"    (proposed family)
#   … etc.
_ROPE_META: Dict[str, Tuple[int, str]] = {
    "none": (0, "RoPE"),
    "linear": (1, "Linear"),
    "ntk": (2, "NTK"),
    "part-ntk": (3, "NTK-by-Parts"),
    "yarn": (4, "YaRN"),
    "my-rope": (5, "My-RoPE"),
    "my-rope-scaled": (6, "My-RoPE+"),
    "my-rope2": (7, "My-RoPE2"),
    "my-rope2-scaled": (8, "My-RoPE2+"),
    "block-layered": (9, "Block-Layered"),
    "block-layered-scaled": (10, "Block-Layered+"),
    "freq-smooth": (11, "Freq-Smooth"),
    "freq-smooth-scaled": (12, "Freq-Smooth+"),
    "freq-reciprocal": (13, "Freq-Reciprocal"),
    "freq-reciprocal-scaled": (14, "Freq-Reciprocal+"),
    "freq-reciprocal-scaled-no-layer": (15, "Freq-Reciprocal (no-layer)"),
    "freq-reciprocal-scaled-adaptive": (0, "Freq-Reciprocal (adaptive)"),
}

# Known default training lengths per model name fragment
_TRAIN_LENGTHS: Dict[str, int] = {
    "llama": 2048,
    "llama2": 4096,
    "mistral": 8192,
    "qwen": 8192,
}


# ============================================================================ #
#  Filename parser                                                              #
# ============================================================================ #


def _parse_filename(path: Path) -> Optional[Dict]:
    """
    Parse a perplexity result filename into structured metadata.

    Handles the three filename formats produced by generate_save_filename:
        - {model}_{rope_type}.json
        - {model}_{rope_type}_dynamic.json
        - {model}_{rope_type}_factor{X_Y}.json

    Args:
        path: Path object pointing to the JSON file.

    Returns:
        A dictionary containing parsed metadata with keys: model_name, rope_type,
        scaling_mode, factor, display_label, color, marker, order_key; or None
        if the filename cannot be parsed.
    """
    stem = path.stem
    sorted_types = sorted(_ROPE_META.keys(), key=len, reverse=True)

    matched_rope = None
    matched_model = None
    matched_suffix = None

    for rope_type in sorted_types:
        pattern = f"_{rope_type}"
        idx = stem.find(pattern)
        if idx == -1:
            continue
        after = stem[idx + len(pattern) :]
        if after == "" or after.startswith("_dynamic") or after.startswith("_factor"):
            matched_model = stem[:idx]
            matched_rope = rope_type
            matched_suffix = after
            break

    if matched_rope is None:
        parts = stem.split("_", 1)
        if len(parts) == 2:
            matched_model, matched_rope = parts
            matched_suffix = ""
        else:
            return None

    factor: Optional[float] = None
    scaling_mode = "none" if matched_rope == "none" else "unknown"

    if matched_suffix == "":
        if matched_rope == "none":
            scaling_mode = "none"
        else:
            scaling_mode = "static"
    elif matched_suffix == "_dynamic":
        scaling_mode = "dynamic"
    elif matched_suffix.startswith("_factor"):
        factor_raw = matched_suffix[len("_factor") :]
        factor_str = factor_raw.replace("_", ".", 1)
        try:
            factor = float(factor_str)
            scaling_mode = "static"
        except ValueError:
            scaling_mode = "static"

    idx_color, base_name = _ROPE_META.get(
        matched_rope,
        (len(_PALETTE) % len(_PALETTE), matched_rope.replace("-", " ").title()),
    )
    color = _PALETTE[idx_color % len(_PALETTE)]
    marker = _MARKERS[idx_color % len(_MARKERS)]

    if scaling_mode == "dynamic":
        suffix_label = " (dyn.)"
    elif factor is not None:
        suffix_label = f" (×{factor:.3g})"
    else:
        suffix_label = ""

    display_label = base_name + suffix_label

    return {
        "path": path,
        "model_name": matched_model,
        "rope_type": matched_rope,
        "scaling_mode": scaling_mode,
        "factor": factor,
        "display_label": display_label,
        "color": color,
        "marker": marker,
        "order_key": (
            (
                _ROPE_DISPLAY_ORDER.index(matched_rope)
                if matched_rope in _ROPE_DISPLAY_ORDER
                else 999
            ),
            0 if scaling_mode == "none" else (1 if scaling_mode == "dynamic" else 2),
            factor or 0.0,
        ),
    }


# ============================================================================ #
#  Data loading                                                                #
# ============================================================================ #


def load_results(paths: List[Path]) -> List[Dict]:
    """
    Load JSON result files and attach parsed metadata.

    Args:
        paths: List of Path objects pointing to JSON result files.

    Returns:
        List of dictionaries, each containing metadata and perplexity data
        from a single result file.
    """
    records = []
    for p in paths:
        meta = _parse_filename(p)
        if meta is None:
            print(f"  [WARN] Cannot parse filename: {p.name} — skipping.")
            continue
        try:
            with open(p, "r") as f:
                data = json.load(f)
            meta["lengths"] = data["lengths"]
            meta["perplexities"] = data["perplexities"]
            records.append(meta)
        except Exception as e:
            print(f"  [WARN] Failed to load {p.name}: {e}")
    return records


def _infer_train_length(model_name: str) -> Optional[int]:
    """
    Infer the original training context length from the model name.

    Args:
        model_name: The model identifier string (e.g., "llama-7b", "mistral-7b").

    Returns:
        The inferred training length in tokens, or 2048 as fallback.
    """
    low = model_name.lower()
    for key, val in sorted(
        _TRAIN_LENGTHS.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if key in low:
            return val
    return 2048  # conservative fallback for LLaMA-family models


# ============================================================================ #
#  Plotting helpers                                                             #
# ============================================================================ #


def _format_length(x: float, _pos=None) -> str:
    """
    Format token counts as human-readable strings.

    Args:
        x: The token count value to format.
        _pos: Position parameter (unused, for matplotlib compatibility).

    Returns:
        Formatted string (e.g., 8192 → '8K', 2048 → '2K').
    """
    x = int(x)
    if x >= 1024 and x % 1024 == 0:
        return f"{x // 1024}K"
    return str(x)


def _draw_train_boundary(ax: plt.Axes, train_length: int) -> None:
    """
    Draw a vertical dashed line at the training context boundary.

    Args:
        ax: Matplotlib Axes object to draw on.
        train_length: The training context length in tokens.

    Returns:
        None
    """
    ax.axvline(
        train_length,
        color="#555555",
        linewidth=1.1,
        linestyle=(0, (4, 3)),  # loosely dashed
        alpha=0.70,
        zorder=1,
        label=f"Training length ({_format_length(train_length)})",
    )


def _clip_perplexity(ppls: List[float], cap: float = 1000.0) -> List[float]:
    """
    Replace diverged perplexity values with NaN for clean plotting.

    Args:
        ppls: List of perplexity values.
        cap: Maximum threshold; values above this are replaced with NaN.

    Returns:
        List of perplexity values with diverged values replaced by NaN.
    """
    return [p if p < cap else float("nan") for p in ppls]


# ============================================================================ #
#  Main plot functions                                                          #
# ============================================================================ #


def plot_perplexity_combined(
    records: List[Dict],
    train_length: Optional[int],
    out_dir: str,
    fmts: List[str],
    log_scale: bool,
    ppl_cap: float,
    figsize: Tuple[float, float],
    dpi: int,
) -> None:
    """
    Plot a single figure comparing the five primary dynamic-scaling methods.

    Includes RoPE (none), Linear, NTK, NTK-by-Parts, and Freq-Reciprocal.
    Only dynamic variants are included for non-baseline methods to ensure
    a fair comparison without manual factor tuning.

    Args:
        records: List of dictionaries containing perplexity data and metadata.
        train_length: The training context length; if None, inferred from model name.
        out_dir: Output directory for saving the plot.
        fmts: List of output file formats (e.g., ['png', 'pdf']).
        log_scale: Whether to use logarithmic scale for y-axis.
        ppl_cap: Maximum perplexity threshold for clipping diverged values.
        figsize: Figure size as (width, height) tuple.
        dpi: Resolution in dots per inch.

    Returns:
        None
    """
    _COMBINED_TYPES = {"none", "linear", "ntk", "part-ntk", "freq-reciprocal"}

    if not records:
        print("  No records to plot.")
        return

    models = sorted({r["model_name"] for r in records})

    for model in models:
        # Keep only the five target types; for non-"none" types use dynamic only
        model_records = [
            r
            for r in records
            if r["model_name"] == model
            and r["rope_type"] in _COMBINED_TYPES
            and (r["rope_type"] == "none" or r["scaling_mode"] == "dynamic")
        ]
        if not model_records:
            print(
                f"  [SKIP] No matching dynamic records for '{model}' — combined plot skipped."
            )
            continue
        model_records.sort(key=lambda r: r["order_key"])

        fig, ax = plt.subplots(figsize=figsize)

        tl = train_length or _infer_train_length(model)
        _draw_train_boundary(ax, tl)

        for rec in model_records:
            lengths = rec["lengths"]
            ppls = _clip_perplexity(rec["perplexities"], ppl_cap)
            # Strip " (dyn.)" — redundant since every non-none curve here is dynamic
            label = rec["display_label"].replace(" (dyn.)", "")
            ax.plot(
                lengths,
                ppls,
                color=rec["color"],
                marker=rec["marker"],
                label=label,
                linewidth=1.9,
                markersize=5.5,
                markeredgewidth=0.6,
                markeredgecolor="white",
                zorder=3,
            )

        ax.set_xlabel("Context Length (tokens)", fontsize=12)
        ax.set_ylabel("Perplexity", fontsize=12)

        model_display = (
            model.upper()
            if "llama" in model.lower()
            else model.replace("-", " ").title()
        )
        ax.set_title(
            f"Perplexity vs. Context Length — {model_display}",
            fontsize=13,
            pad=9,
        )

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_length))
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))

        if log_scale:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.tick_params(which="minor", length=2)

        leg = ax.legend(
            loc="upper right",
            ncol=3,
            framealpha=0.95,
        )
        leg.get_frame().set_linewidth(0.6)

        ax.set_xlim(left=0, right=max(lengths) * 1.08)
        non_baseline_ppls = [
            p
            for r in model_records
            if r["rope_type"] != "none"
            for p in r["perplexities"]
            if p < ppl_cap
        ]
        if non_baseline_ppls:
            y_max = max(non_baseline_ppls) * 1.3
            ax.set_ylim(bottom=0, top=y_max)
        else:
            _set_ylim(ax, model_records, ppl_cap, log_scale)

        fig.tight_layout()
        _save(fig, out_dir, "perplexity_dynamic_comparison", fmts, dpi)
        plt.close(fig)


def plot_perplexity_by_family(
    records: List[Dict],
    train_length: Optional[int],
    out_dir: str,
    fmts: List[str],
    log_scale: bool,
    ppl_cap: float,
    dpi: int,
) -> None:
    """
    Plot a multi-panel figure with one subplot per RoPE family.

    Each panel shows the family's curves against the Standard RoPE baseline.
    Useful for comparing static vs dynamic scaling within each family.

    Args:
        records: List of dictionaries containing perplexity data and metadata.
        train_length: The training context length; if None, inferred from model name.
        out_dir: Output directory for saving the plot.
        fmts: List of output file formats (e.g., ['png', 'pdf']).
        log_scale: Whether to use logarithmic scale for y-axis.
        ppl_cap: Maximum perplexity threshold for clipping diverged values.
        dpi: Resolution in dots per inch.

    Returns:
        None
    """
    baseline_rope = "none"
    baselines = {r["model_name"]: r for r in records if r["rope_type"] == baseline_rope}
    non_baselines = [r for r in records if r["rope_type"] != baseline_rope]

    if not non_baselines:
        print("  No non-baseline records found for family comparison plot.")
        return

    models = sorted({r["model_name"] for r in records})

    for model in models:
        model_non_bl = [r for r in non_baselines if r["model_name"] == model]
        if not model_non_bl:
            continue

        families = sorted(
            {r["rope_type"] for r in model_non_bl},
            key=lambda t: (
                _ROPE_DISPLAY_ORDER.index(t) if t in _ROPE_DISPLAY_ORDER else 999
            ),
        )

        n = len(families)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.5 * ncols, 3.8 * nrows),
            squeeze=False,
            sharey=False,
        )

        tl = train_length or _infer_train_length(model)
        baseline = baselines.get(model)

        for panel_idx, rope_type in enumerate(families):
            row, col = divmod(panel_idx, ncols)
            ax = axes[row][col]

            _draw_train_boundary(ax, tl)

            # Draw baseline first (grey, thin)
            if baseline is not None:
                ax.plot(
                    baseline["lengths"],
                    _clip_perplexity(baseline["perplexities"], ppl_cap),
                    color="#999999",
                    marker="o",
                    linewidth=1.3,
                    markersize=3.5,
                    linestyle="--",
                    label=baseline["display_label"],
                    zorder=2,
                )

            family_recs = [r for r in model_non_bl if r["rope_type"] == rope_type]
            family_recs.sort(key=lambda r: r["order_key"])

            for rec in family_recs:
                ax.plot(
                    rec["lengths"],
                    _clip_perplexity(rec["perplexities"], ppl_cap),
                    color=rec["color"],
                    marker=rec["marker"],
                    label=rec["display_label"],
                    linewidth=1.9,
                    markersize=5,
                    markeredgewidth=0.5,
                    markeredgecolor="white",
                    zorder=3,
                )

            _, base_name = _ROPE_META.get(
                rope_type,
                (0, rope_type.replace("-", " ").title()),
            )
            ax.set_title(base_name, fontsize=11, pad=5)
            ax.set_xlabel("Context Length", fontsize=10)
            ax.set_ylabel("Perplexity ↓", fontsize=10)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_length))
            ax.tick_params(which="both", direction="in", top=True, right=True)
            if log_scale:
                ax.set_yscale("log")
            ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
            ax.set_xlim(left=0)
            _set_ylim(
                ax,
                family_recs + ([baseline] if baseline else []),
                ppl_cap,
                log_scale,
            )

        # Hide unused subplots
        for extra in range(n, nrows * ncols):
            r, c = divmod(extra, ncols)
            axes[r][c].set_visible(False)

        model_display = model.replace("-", " ").title()
        fig.suptitle(
            f"Perplexity by RoPE Family — {model_display}",
            fontsize=14,
            y=1.01,
        )
        fig.tight_layout()
        _save(fig, out_dir, "perplexity_by_family", fmts, dpi)
        plt.close(fig)


def plot_perplexity_delta(
    records: List[Dict],
    train_length: Optional[int],
    out_dir: str,
    fmts: List[str],
    ppl_cap: float,
    dpi: int,
) -> None:
    """
    Plot delta-perplexity: PPL(method) - PPL(Standard RoPE) vs context length.

    Values below zero indicate the method outperforms Standard RoPE at that length.
    This visualization highlights extrapolation gains without absolute scale dominance.

    Args:
        records: List of dictionaries containing perplexity data and metadata.
        train_length: The training context length; if None, inferred from model name.
        out_dir: Output directory for saving the plot.
        fmts: List of output file formats (e.g., ['png', 'pdf']).
        ppl_cap: Maximum perplexity threshold for clipping diverged values.
        dpi: Resolution in dots per inch.

    Returns:
        None
    """
    models = sorted({r["model_name"] for r in records})

    for model in models:
        model_records = [r for r in records if r["model_name"] == model]
        baseline = next((r for r in model_records if r["rope_type"] == "none"), None)
        if baseline is None:
            print(
                f"  [SKIP] No Standard RoPE baseline for '{model}' — delta plot skipped."
            )
            continue

        non_baselines = [r for r in model_records if r["rope_type"] != "none"]
        if not non_baselines:
            continue

        fig, ax = plt.subplots(figsize=(7.0, 4.2))

        tl = train_length or _infer_train_length(model)
        _draw_train_boundary(ax, tl)
        ax.axhline(
            0,
            color="#333333",
            linewidth=0.9,
            linestyle="-",
            zorder=1,
            label="RoPE (reference)",
        )

        bl_interp = dict(zip(baseline["lengths"], baseline["perplexities"]))

        for rec in sorted(non_baselines, key=lambda r: r["order_key"]):
            xs, deltas = [], []
            for length, ppl in zip(rec["lengths"], rec["perplexities"]):
                if ppl >= ppl_cap:
                    continue
                bl_ppl = bl_interp.get(length)
                if bl_ppl is None or bl_ppl >= ppl_cap:
                    continue
                xs.append(length)
                deltas.append(ppl - bl_ppl)

            if not xs:
                continue

            ax.plot(
                xs,
                deltas,
                color=rec["color"],
                marker=rec["marker"],
                label=rec["display_label"],
                linewidth=1.8,
                markersize=5,
                markeredgewidth=0.5,
                markeredgecolor="white",
                zorder=3,
            )

        ax.set_xlabel("Context Length (tokens)", fontsize=12)
        ax.set_ylabel("ΔPPL vs RoPE ↓", fontsize=12)

        model_display = (
            model.upper()
            if "llama" in model.lower()
            else model.replace("-", " ").title()
        )
        ax.set_title(
            f"Perplexity Gain over RoPE — {model_display}\n"
            "Negative values indicate improvement",
            fontsize=12,
            pad=8,
        )

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_length))
        ax.tick_params(which="both", direction="in", top=True, right=True)

        n_curves = len(non_baselines) + 2
        if n_curves > 8:
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0,
                fontsize=9,
            )
        else:
            ax.legend(loc="lower left", fontsize=9)

        ax.set_xlim(left=0)
        fig.tight_layout()
        _save(fig, out_dir, "perplexity_delta", fmts, dpi)
        plt.close(fig)


# ============================================================================ #
#  Helpers                                                                     #
# ============================================================================ #


def _set_ylim(
    ax: plt.Axes,
    records: List[Dict],
    ppl_cap: float,
    log_scale: bool,
) -> None:
    """
    Set sensible y-axis limits, ignoring diverged runs.

    Args:
        ax: Matplotlib Axes object to modify.
        records: List of dictionaries containing perplexity data.
        ppl_cap: Maximum perplexity threshold; values above are ignored.
        log_scale: Whether the y-axis uses logarithmic scale.

    Returns:
        None
    """
    all_ppls = [
        p
        for r in records
        for p in r.get("perplexities", [])
        if p < ppl_cap and not np.isnan(p)
    ]
    if not all_ppls:
        return
    lo, hi = min(all_ppls), max(all_ppls)
    margin = 0.08 * (hi - lo) if not log_scale else 0.0
    if log_scale:
        ax.set_ylim(lo * 0.92, hi * 1.25)
    else:
        ax.set_ylim(max(0, lo - margin), hi + margin * 2)


def _save(
    fig: plt.Figure,
    out_dir: str,
    name: str,
    fmts: List[str],
    dpi: int,
) -> None:
    """
    Save a figure to multiple formats.

    Args:
        fig: Matplotlib Figure object to save.
        out_dir: Output directory path.
        name: Base filename without extension.
        fmts: List of output formats (e.g., ['png', 'pdf']).
        dpi: Resolution in dots per inch.

    Returns:
        None
    """
    os.makedirs(out_dir, exist_ok=True)
    for fmt in fmts:
        path = os.path.join(out_dir, f"{name}.{fmt}")
        fig.savefig(path, dpi=dpi, format=fmt)
        print(f"  saved → {path}")


# ============================================================================ #
#  CLI                                                                         #
# ============================================================================ #


def _build_parser() -> argparse.ArgumentParser:
    """
    Build and configure the argument parser for the CLI.

    Returns:
        Configured ArgumentParser instance.
    """
    p = argparse.ArgumentParser(
        description="Paper-quality perplexity plots for RoPE context-extension experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--result-dir",
        "-r",
        type=str,
        default="results/perplexity",
        help="Directory containing perplexity JSON files (default: results/perplexity).",
    )
    p.add_argument(
        "--files",
        "-f",
        type=str,
        nargs="*",
        default=None,
        help="Explicit list of JSON files to plot (overrides --result-dir).",
    )
    p.add_argument(
        "--out-dir",
        "-o",
        type=str,
        default="drawer/perplexity",
        help="Output directory for figures (default: drawer/perplexity).",
    )
    p.add_argument(
        "--fmt",
        type=str,
        nargs="+",
        default=["png"],
        choices=["png", "pdf", "svg", "eps"],
        help="Output format(s) (default: png). Use 'pdf png' for LaTeX + preview.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster formats (default: 300).",
    )
    p.add_argument(
        "--train-length",
        type=int,
        default=None,
        help="Override the inferred training context length for the boundary line.",
    )
    p.add_argument(
        "--ppl-cap",
        type=float,
        default=1000.0,
        help="Perplexity values above this cap are treated as diverged and hidden (default: 1000).",
    )
    p.add_argument(
        "--log-scale",
        action="store_true",
        default=False,
        help="Use logarithmic y-axis.",
    )
    p.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[7.0, 4.2],
        metavar=("W", "H"),
        help="Figure size in inches for the combined plot (default: 7.0 4.2).",
    )
    p.add_argument(
        "--no-family",
        action="store_true",
        default=False,
        help="Skip the per-family multi-panel plot.",
    )
    p.add_argument(
        "--no-delta",
        action="store_true",
        default=False,
        help="Skip the ΔPPL plot (requires a Standard RoPE baseline file).",
    )
    return p


def main() -> None:
    """
    Main entry point for the perplexity plotting CLI.

    Parses command-line arguments, loads perplexity data from JSON files,
    and generates comparison plots for RoPE extension methods.

    Returns:
        None
    """
    args = _build_parser().parse_args()

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        result_dir = Path(args.result_dir)
        if not result_dir.exists():
            sys.exit(f"[ERROR] Result directory not found: {result_dir}")
        paths = sorted(result_dir.glob("*.json"))
        if not paths:
            sys.exit(f"[ERROR] No JSON files found in {result_dir}")

    print(f"\nFound {len(paths)} file(s):")
    for p in paths:
        print(f"  {p.name}")

    records = load_results(paths)
    if not records:
        sys.exit("[ERROR] No valid records loaded.")

    print(f"\nLoaded {len(records)} record(s):")
    for r in records:
        label = r["display_label"]
        n_pts = len(r["lengths"])
        l_min = _format_length(r["lengths"][0])
        l_max = _format_length(r["lengths"][-1])
        p_min = min(r["perplexities"])
        p_max = max(r["perplexities"])
        print(
            f"  [{r['model_name']}]  {label:<35s}  "
            f"lengths: {l_min}–{l_max} ({n_pts} pts)  "
            f"PPL: {p_min:.2f}–{p_max:.2f}"
        )

    print(f"\nGenerating figures → {args.out_dir}/")

    print("\n[1/3] Combined overlay plot")
    plot_perplexity_combined(
        records,
        train_length=args.train_length,
        out_dir=args.out_dir,
        fmts=args.fmt,
        log_scale=args.log_scale,
        ppl_cap=args.ppl_cap,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
    )

    if not args.no_family:
        print("\n[2/3] Per-family multi-panel plot")
        plot_perplexity_by_family(
            records,
            train_length=args.train_length,
            out_dir=args.out_dir,
            fmts=args.fmt,
            log_scale=args.log_scale,
            ppl_cap=args.ppl_cap,
            dpi=args.dpi,
        )
    else:
        print("\n[2/3] Skipped (--no-family).")

    if not args.no_delta:
        print("\n[3/3] Delta-PPL plot")
        plot_perplexity_delta(
            records,
            train_length=args.train_length,
            out_dir=args.out_dir,
            fmts=args.fmt,
            ppl_cap=args.ppl_cap,
            dpi=args.dpi,
        )
    else:
        print("\n[3/3] Skipped (--no-delta).")

    print("\nAll done.")


if __name__ == "__main__":
    main()
