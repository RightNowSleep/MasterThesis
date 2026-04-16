"""Runtime comparison bar chart generator for context extension methods.

Reads performance JSON files from the results directory and produces
two bar charts -- one in English and one in Chinese -- comparing the
total runtime (seconds) of each RoPE context extension method.

Method name mapping:
    none                     -> RoPE
    linear                   -> PI
    ntk                      -> NTK-aware
    part-ntk                 -> NTK-by-parts
    yarn                     -> YaRN
    inverse-dual-rope-scaled -> BS2

Usage:
    python drawer/runtime.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Palatino"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.45,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

_RESULT_DIR = Path(__file__).resolve().parent.parent / "results" / "performance"
_OUT_DIR = Path(__file__).resolve().parent / "runtime"

_METHOD_MAP: Dict[str, str] = {
    "none": "RoPE",
    "linear": "PI",
    "ntk": "NTK-aware",
    "part-ntk": "NTK-by-parts",
    "yarn": "YaRN",
    "inverse-dual-rope-scaled": "BS2",
}

_METHOD_MAP_CN: Dict[str, str] = {
    "none": "RoPE",
    "linear": "PI",
    "ntk": "NTK-aware",
    "part-ntk": "NTK-by-parts",
    "yarn": "YaRN",
    "inverse-dual-rope-scaled": "BS2",
}

_PALETTE: Dict[str, str] = {
    "none": "#2166AC",
    "linear": "#D6604D",
    "ntk": "#4DAF4A",
    "part-ntk": "#984EA3",
    "yarn": "#FF7F00",
    "inverse-dual-rope-scaled": "#A65628",
}

_DISPLAY_ORDER: List[str] = [
    "none",
    "linear",
    "ntk",
    "part-ntk",
    "yarn",
    "inverse-dual-rope-scaled",
]


def _load_runtimes(result_dir: Path) -> List[Tuple[str, str, float]]:
    """Load total_runtime_seconds from each performance JSON file.

    Scans the result directory for JSON files matching the known method
    names and extracts the ``total_runtime_seconds`` field.

    Args:
        result_dir: Path to the directory containing performance JSON files.

    Returns:
        List of (rope_type, display_name, runtime_seconds) tuples sorted
        by the canonical display order.
    """
    records: List[Tuple[str, str, float]] = []
    for rope_type in _DISPLAY_ORDER:
        candidates = sorted(result_dir.glob(f"*_{rope_type}_dynamic.json"))
        if rope_type == "none":
            candidates = sorted(result_dir.glob(f"*_{rope_type}.json"))
        if not candidates:
            print(f"  [WARN] No result file found for '{rope_type}' -- skipping.")
            continue
        fp = candidates[0]
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            runtime = data["total_runtime_seconds"]
            records.append((rope_type, _METHOD_MAP[rope_type], runtime))
        except Exception as e:
            print(f"  [WARN] Failed to load {fp.name}: {e}")
    return records


def _plot_bar(
    records: List[Tuple[str, str, float]],
    lang: str,
    out_dir: Path,
    dpi: int = 300,
) -> None:
    """Draw and save a bar chart for runtime comparison.

    Args:
        records: List of (rope_type, display_name, runtime_seconds) tuples.
        lang: 'en' for English or 'cn' for Chinese.
        out_dir: Output directory for the saved figure.
        dpi: Resolution in dots per inch.
    """
    if lang == "cn":
        plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["font.family"] = "sans-serif"

    labels = [r[1] for r in records]
    values = [r[2] for r in records]
    colors = [_PALETTE[r[0]] for r in records]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        values,
        width=0.55,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )

    for bar, val in zip(bars, values):
        ax.annotate(
            f"{val:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)

    if lang == "en":
        ax.set_xlabel("Method", fontsize=12)
        ax.set_ylabel("Runtime (seconds)", fontsize=12)
        ax.set_title(
            "Runtime Comparison of Context Extension Methods",
            fontsize=13,
            pad=10,
        )
        fname = "runtime_comparison_en"
    else:
        ax.set_xlabel("方法", fontsize=12)
        ax.set_ylabel("运行时间（秒）", fontsize=12)
        fname = "runtime_comparison_cn"

    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax.set_ylim(bottom=0, top=max(values) * 1.15)
    ax.tick_params(which="both", direction="in", top=False, right=False)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{fname}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / fname}.png")


def main() -> None:
    """Entry point: load data and generate both English and Chinese charts."""
    print(f"Loading performance data from: {_RESULT_DIR}")
    records = _load_runtimes(_RESULT_DIR)
    if not records:
        print("  No data found. Exiting.")
        return

    print(f"  Found {len(records)} methods:")
    for rope_type, name, rt in records:
        print(f"    {name:15s}  {rt:>10.2f} s")

    _plot_bar(records, lang="en", out_dir=_OUT_DIR)
    _plot_bar(records, lang="cn", out_dir=_OUT_DIR)


if __name__ == "__main__":
    main()
