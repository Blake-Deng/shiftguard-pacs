#!/usr/bin/env python3
"""Generate the quantitative 2x2 evidence figure from committed summaries."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image


SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE
OUT = SOURCE / "figures"
SUMMARY_PATH = ROOT / "revision" / "cgc_v2_summaries" / "complete_cgc_summary.json"
BASELINE_SUMMARY_PATH = ROOT / "revision" / "summaries" / "all_methods_five_seed_summary.json"

COLORS = {
    "strong_aug": "#465564",
    "feature_plus_kl": "#177EAA",
    "cgc": "#D1495B",
    "mixstyle": "#3A8D6D",
    "swad_epoch": "#8064A2",
    "positive": "#16825D",
    "negative": "#C43C4E",
    "neutral": "#727C86",
    "grid": "#DCE2E7",
    "ink": "#20272E",
}

LABELS = {
    "strong_aug": "Strong Aug.",
    "feature_plus_kl": "Feature+KL",
    "cgc": "CGC",
    "mixstyle": "MixStyle",
    "swad_epoch": "SWAD",
}

DATASET_LABELS = {"pacs": "PACS", "vlcs": "VLCS", "officehome": "OfficeHome"}
TARGETS = ["Photo", "Art_Painting", "Cartoon", "Sketch"]
TARGET_DIRS = {
    "Photo": "photo",
    "Art_Painting": "art_painting",
    "Cartoon": "cartoon",
    "Sketch": "sketch",
}

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 12.0,
    "axes.labelsize": 10.2,
    "xtick.labelsize": 9.1,
    "ytick.labelsize": 9.1,
    "legend.fontsize": 8.7,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def style_axis(ax, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.75, alpha=0.9)
    ax.set_axisbelow(True)


def paired_feature_minus_aug(block: dict) -> dict:
    feature = block["feature_plus_kl"]["macro_by_seed"]
    strong = block["strong_aug"]["macro_by_seed"]
    values = np.array([feature[key] - strong[key] for key in strong], dtype=float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    # t(0.975, 4) for the five predeclared seeds.
    half = 2.7764451051977987 * sd / np.sqrt(5)
    return {"mean": mean, "ci95": [mean - half, mean + half], "deltas": values.tolist()}


def make_evidence_figure(summary: dict) -> None:
    datasets = summary["resnet50"]
    baseline_summary = json.loads(BASELINE_SUMMARY_PATH.read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(14.2, 8.25), constrained_layout=False)
    gs = fig.add_gridspec(
        2, 2, left=0.065, right=0.985, bottom=0.085, top=0.955,
        wspace=0.30, hspace=0.48,
    )

    # (a) Residual accuracy relative to the matched augmentation baseline.
    ax = fig.add_subplot(gs[0, 0])
    methods = ["feature_plus_kl", "cgc", "mixstyle", "swad_epoch"]
    x = np.arange(3)
    width = 0.18
    for index, method in enumerate(methods):
        values = []
        for dataset in ("pacs", "vlcs", "officehome"):
            block = datasets[dataset]
            if method in block:
                method_mean = block[method]["macro"]["mean"]
            else:
                method_mean = baseline_summary["datasets"][dataset]["methods"][method]["macro_mean"]
            values.append(method_mean - block["strong_aug"]["macro"]["mean"])
        positions = x + (index - 1.5) * width
        bars = ax.bar(
            positions, values, width=width, color=COLORS[method], label=LABELS[method],
            edgecolor="white", linewidth=0.7, zorder=3,
        )
        for bar, value in zip(bars, values):
            offset = 0.10 if value >= 0 else -0.10
            ax.text(
                bar.get_x() + bar.get_width() / 2, value + offset, f"{value:+.2f}",
                ha="center", va="bottom" if value >= 0 else "top", fontsize=7.8,
                fontweight="bold", color=COLORS["ink"],
            )
    ax.axhline(0, color=COLORS["ink"], linewidth=1.0)
    ax.set_xticks(x, ["PACS", "VLCS", "OfficeHome"])
    ax.set_ylim(-3.55, 2.35)
    ax.set_ylabel("Macro change from Strong Aug. (pp)")
    ax.set_title("(a) Residual benefit beyond strong augmentation", loc="left", fontweight="bold")
    style_axis(ax)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.50, -0.18), ncol=4,
        frameon=False, handlelength=1.25, columnspacing=1.15,
    )

    # (b) Paired seed-level uncertainty for both consistency objectives.
    ax = fig.add_subplot(gs[0, 1])
    scenarios = [
        ("PACS / ResNet-50", datasets["pacs"]),
        ("VLCS / ResNet-50", datasets["vlcs"]),
        ("OfficeHome / ResNet-50", datasets["officehome"]),
        ("PACS / ViT-S/16", summary["vit_small_patch16_224"]),
    ]
    y = np.arange(len(scenarios))[::-1]
    offsets = {"feature_plus_kl": 0.14, "cgc": -0.14}
    for band_y in y[::2]:
        ax.axhspan(band_y - 0.43, band_y + 0.43, color="#F2F5F7", zorder=0)
    for method in ("feature_plus_kl", "cgc"):
        for yy, (_, block) in zip(y, scenarios):
            result = (
                paired_feature_minus_aug(block)
                if method == "feature_plus_kl"
                else block["paired_cgc_minus_strong_aug"]
            )
            mean = result["mean"]
            lo, hi = result["ci95"]
            row_y = yy + offsets[method]
            seed_values = (result["deltas"] if "deltas" in result
                           else list(result["deltas_by_seed"].values()))
            seed_jitter = np.linspace(-0.065, 0.065, len(seed_values))
            ax.scatter(seed_values, row_y + seed_jitter, s=17, facecolor="white",
                       edgecolor=COLORS[method], linewidth=0.8, alpha=0.85, zorder=2)
            ax.errorbar(
                mean, row_y, xerr=[[mean - lo], [hi - mean]],
                fmt="D" if method == "feature_plus_kl" else "o",
                markersize=6.0, color=COLORS[method], ecolor=COLORS[method],
                elinewidth=2.0, capsize=3.0, capthick=1.2,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3,
            )
            ax.text(
                2.43, row_y, f"{mean:+.2f}", ha="left", va="center",
                color=COLORS[method], fontsize=8.0, fontweight="bold",
            )
    ax.axvline(0, color=COLORS["ink"], linewidth=1.0, zorder=1)
    ax.axvline(2.30, color="#D8DEE3", linewidth=0.8, zorder=1)
    ax.set_yticks(y, [label for label, _ in scenarios])
    ax.set_xlim(-2.15, 2.82)
    ax.set_ylim(-0.55, 3.55)
    ax.set_xlabel("Paired macro difference from Strong Aug. (pp)")
    ax.set_title("(b) Paired uncertainty across five seeds", loc="left", fontweight="bold")
    style_axis(ax, "x")
    ax.text(2.43, 3.42, "Mean", ha="left", va="center", fontsize=7.8,
            color="#68737D", fontweight="bold")
    ax.scatter([], [], marker="D", color=COLORS["feature_plus_kl"], label="Feature+KL")
    ax.scatter([], [], marker="o", color=COLORS["cgc"], label="CGC")
    ax.legend(loc="lower left", frameon=False, ncol=2)

    # (c) Negative-transfer rate, where a run underperforms its paired control.
    ax = fig.add_subplot(gs[1, 0])
    ntr = summary["negative_transfer_rate"]
    groups = ["PACS", "VLCS", "OfficeHome", "Overall"]
    feature = [100 * ntr["feature_plus_kl"][key]["rate"] for key in ("pacs", "vlcs", "officehome", "overall")]
    cgc = [100 * ntr["cgc"][key]["rate"] for key in ("pacs", "vlcs", "officehome", "overall")]
    xx = np.arange(4)
    width = 0.32
    bars_f = ax.bar(xx - width / 2, feature, width, color=COLORS["feature_plus_kl"], label="Feature+KL")
    bars_c = ax.bar(xx + width / 2, cgc, width, color=COLORS["cgc"], label="CGC")
    for bars in (bars_f, bars_c):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                fontsize=8.3, fontweight="bold", color=COLORS["ink"],
            )
    ax.set_xticks(xx, groups)
    ax.set_ylim(0, 64)
    ax.set_ylabel("Negative-transfer rate (%)")
    ax.set_title("(c) Paired negative transfer across formal runs", loc="left", fontweight="bold")
    style_axis(ax)
    ax.legend(loc="upper left", frameon=False, ncol=2)

    # (d) Strict nested source-domain selection.
    ax = fig.add_subplot(gs[1, 1])
    nested = summary["strict_nested_pacs"]
    targets = nested["outer_targets"]
    candidate_methods = ("strong_aug", "feature_plus_kl", "cgc")
    candidate_x = np.arange(3)
    target_colors = ["#2A9D8F", "#E76F51", "#6C5CE7", "#E0A12B"]
    annotation_offsets = {
        "Photo": (0.10, 0.38),
        "Art_Painting": (-0.28, 0.42),
        "Cartoon": (-0.30, -0.48),
        "Sketch": (-0.08, 0.38),
    }
    for column in candidate_x:
        ax.axvspan(column - 0.13, column + 0.13, color="#F1F4F6", zorder=0)
    for row, color in zip(targets, target_colors):
        scores = np.array([row["candidate_inner_scores"][method] for method in candidate_methods])
        selected_index = int(np.argmax(scores))
        target_mean = row["formal_target"]["mean"]
        target_label = row["outer_target"].replace("_", " ")
        ax.plot(candidate_x, scores, color=color, linewidth=2.35, marker="o",
                markersize=5.8, markerfacecolor="white", markeredgewidth=1.5,
                label=target_label, zorder=3)
        ax.scatter(selected_index, scores[selected_index], marker="*", s=155,
                   facecolor=color, edgecolor="white", linewidth=0.9, zorder=5)
        dx, dy = annotation_offsets[row["outer_target"]]
        ax.annotate(
            f"{target_mean:.2f}%", xy=(selected_index, scores[selected_index]),
            xytext=(selected_index + dx, scores[selected_index] + dy),
            ha="center", va="center", fontsize=7.9, fontweight="bold", color=color,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white",
                  "edgecolor": color, "linewidth": 0.8},
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.9,
                        "shrinkA": 2, "shrinkB": 5},
        )
    ax.set_xticks(candidate_x, ["Strong Aug.", "Feature+KL", "CGC"])
    ax.set_xlim(-0.28, 2.28)
    ax.set_ylim(81.65, 86.18)
    ax.set_ylabel("Inner source-domain validation (%)")
    ax.set_title("(d) Nested rankings change with the outer target", loc="left", fontweight="bold")
    style_axis(ax)
    ax.text(0.985, 0.965, f"Nested macro {nested['nested_macro']['mean']:.2f} +/- "
            f"{nested['nested_macro']['sample_sd']:.2f}%", transform=ax.transAxes,
            ha="right", va="top", fontsize=8.1, color="#5D6872", fontweight="bold")
    ax.text(0.02, 0.035, "Stars: source-only selected objective; labels: held-out accuracy",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.7, color="#5D6872")
    ax.legend(loc="upper center", bbox_to_anchor=(0.50, -0.17), ncol=4,
              frameon=False, handlelength=1.5, columnspacing=1.0)

    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"fig2_cgc_evidence.{suffix}", dpi=360, bbox_inches="tight", facecolor="white")
    plt.close(fig)



def main() -> None:
    make_evidence_figure(load_summary())
    print(OUT / "fig2_cgc_evidence.pdf")


if __name__ == "__main__":
    main()
