#!/usr/bin/env python3
"""Render image- and patient-level DCA in DRCP/Elsevier artwork format."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tables" / "decision_curve_pointwise.csv"
FIGURES = ROOT / "figures"

COLORS = {
    "mg": "#2878B5",
    "model_v": "#777777",
    "all": "#009E73",
    "none": "#111111",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_panel(
    axis: plt.Axes, frame: pd.DataFrame, label: str, title: str
) -> None:
    threshold = frame["threshold_probability"]
    axis.fill_between(
        threshold,
        frame["mg_ci95_low"],
        frame["mg_ci95_high"],
        color=COLORS["mg"],
        alpha=0.12,
        linewidth=0,
    )
    axis.fill_between(
        threshold,
        frame["model_v_ci95_low"],
        frame["model_v_ci95_high"],
        color=COLORS["model_v"],
        alpha=0.12,
        linewidth=0,
    )
    axis.plot(
        threshold,
        frame["mg_rl_net_benefit"],
        color=COLORS["mg"],
        label="MG_RL",
    )
    axis.plot(
        threshold,
        frame["model_v_net_benefit"],
        color=COLORS["model_v"],
        label="Model V fixed referral",
    )
    axis.plot(
        threshold,
        frame["refer_all_net_benefit"],
        color=COLORS["all"],
        linestyle="--",
        linewidth=1.1,
        label="Refer all",
    )
    axis.plot(
        threshold,
        frame["refer_none_net_benefit"],
        color=COLORS["none"],
        linewidth=0.9,
        label="Refer none",
    )
    axis.set_title(f"{label}  {title}", loc="left", fontweight="bold", pad=5)
    axis.set_xlim(0.0, 0.50)
    axis.set_ylim(-0.04, 0.13)
    axis.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    axis.set_yticks([-0.04, 0.0, 0.04, 0.08, 0.12])
    axis.set_xlabel("Threshold probability")
    axis.axhline(0, color=COLORS["none"], linewidth=0.7)
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=3, width=0.65)


def main() -> None:
    configure_style()
    data = pd.read_csv(SOURCE)
    FIGURES.mkdir(exist_ok=True)
    fig, axes = plt.subplots(
        1, 2, figsize=(7.48, 3.05), sharey=True
    )
    draw_panel(
        axes[0], data[data["level"] == "image"], "A", "Image level"
    )
    draw_panel(
        axes[1], data[data["level"] == "patient"], "B", "Patient level"
    )
    axes[0].set_ylabel("Net benefit")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        handlelength=2.8,
        columnspacing=1.5,
    )
    fig.subplots_adjust(
        left=0.075, right=0.99, bottom=0.17, top=0.82, wspace=0.22
    )
    base = FIGURES / "Figure_Decision_Curve_MG_RL_vs_Model_V"
    fig.savefig(
        base.with_suffix(".pdf"),
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    fig.savefig(
        base.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=1000,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    with Image.open(base.with_suffix(".tiff")) as image:
        rgb_image = image.convert("RGB")
        rgb_image.save(
            base.with_suffix(".tiff"),
            format="TIFF",
            dpi=(1000, 1000),
            compression="tiff_lzw",
        )
    with Image.open(base.with_suffix(".tiff")) as image:
        if min(image.info.get("dpi", (0, 0))) < 999:
            raise RuntimeError("TIFF resolution check failed")
        if image.mode != "RGB":
            raise RuntimeError(f"Unexpected TIFF mode: {image.mode}")


if __name__ == "__main__":
    main()
