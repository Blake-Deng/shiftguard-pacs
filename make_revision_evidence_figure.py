#!/usr/bin/env python3
"""Create the compact 2x2 evidence figure used by the revised paper."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
SUMMARY = ROOT / "revision" / "summaries"
COLORS = {
    "erm": "#626B75",
    "strong_aug": "#1976B9",
    "feature_plus_kl": "#D94733",
    "mixstyle": "#16835F",
    "swad_epoch": "#F0A12B",
}

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.0,
    "axes.titlesize": 11.7,
    "axes.labelsize": 10.0,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "legend.fontsize": 8.7,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def style(ax, axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, color="#DDE2E7", linewidth=0.75, alpha=0.9)
    ax.set_axisbelow(True)


def load_json(name: str):
    return json.loads((SUMMARY / name).read_text(encoding="utf-8"))


def main() -> None:
    all_methods = load_json("all_methods_five_seed_summary.json")
    pacs = load_json("pacs_five_seed_summary.json")
    cross = load_json("cross_dataset_five_seed_summary.json")
    vit = load_json("vit_direct_five_seed_summary.json")
    with (SUMMARY / "compact_sensitivity_table.csv").open(newline="", encoding="utf-8") as handle:
        sensitivity = list(csv.DictReader(handle))

    fig = plt.figure(figsize=(13.8, 8.05), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, left=0.065, right=0.985, bottom=0.075, top=0.965,
                          wspace=0.29, hspace=0.50)

    # (a) Multi-benchmark macro comparison. PACS ERM is omitted because only
    # three seeds exist; the paper reports it separately with that disclosure.
    ax = fig.add_subplot(gs[0, 0])
    datasets = ["pacs", "vlcs", "officehome"]
    dataset_labels = ["PACS", "VLCS", "OfficeHome"]
    methods = ["strong_aug", "feature_plus_kl", "mixstyle", "swad_epoch"]
    labels = ["Strong Aug.", "Feature + KL", "MixStyle", "SWAD (epoch)"]
    x = np.arange(3)
    width = 0.19
    for index, (method, label) in enumerate(zip(methods, labels)):
        means = [all_methods["datasets"][dataset]["methods"][method]["macro_mean"]
                 for dataset in datasets]
        stds = [all_methods["datasets"][dataset]["methods"][method]["macro_sample_sd"]
                for dataset in datasets]
        pos = x + (index - 1.5) * width
        bars = ax.bar(pos, means, width, yerr=stds, capsize=2.3, label=label,
                      color=COLORS[method], edgecolor="white", linewidth=0.6,
                      error_kw={"elinewidth": 0.9, "capthick": 0.9})
        for bar, value in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.7, f"{value:.1f}",
                    ha="center", va="bottom", rotation=90, fontsize=7.6,
                    fontweight="bold", color="#293038")
    ax.set_xticks(x, dataset_labels)
    ax.set_ylim(60, 91.5)
    ax.set_ylabel("Five-seed macro accuracy (%)")
    ax.set_title("(a) Multi-benchmark results under one schedule", loc="left", fontweight="bold")
    style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4,
              frameon=False, columnspacing=1.0, handlelength=1.2)

    # (b) Paired residual effect and 95% t intervals.
    ax = fig.add_subplot(gs[0, 1])
    paired = [
        ("PACS / ResNet-50", pacs["paired_feature_plus_kl_minus_strong_aug"]),
        ("VLCS / ResNet-50", cross["datasets"]["vlcs"]["paired_feature_plus_kl_minus_strong_aug"]),
        ("OfficeHome / ResNet-50", cross["datasets"]["officehome"]["paired_feature_plus_kl_minus_strong_aug"]),
        ("PACS / ViT-S/16", vit["paired_feature_plus_kl_minus_strong_aug"]),
    ]
    ypos = np.arange(len(paired))[::-1]
    ax.axvspan(-1.35, 0, color="#D94733", alpha=0.045, zorder=0)
    ax.axvspan(0, 2.75, color="#16835F", alpha=0.045, zorder=0)
    seed_colors = ["#1976B9", "#F0A12B", "#16835F", "#7A5195", "#6C757D"]
    jitter = np.linspace(-0.13, 0.13, 5)
    for y, (label, result) in zip(ypos, paired):
        mean = result["mean_delta"]
        lo, hi = result["ci95"]
        seed_delta = list(result["delta_by_seed"].values())
        ax.plot([lo, hi], [y, y], color="#525D68", lw=3.0, solid_capstyle="round")
        for value, dy, color in zip(seed_delta, jitter, seed_colors):
            ax.scatter(value, y + dy, s=32, color=color, edgecolor="white",
                       linewidth=0.7, alpha=0.92, zorder=3)
        ax.scatter(mean, y, s=95, marker="D", color=COLORS["feature_plus_kl"],
                   edgecolor="white", linewidth=1.2, zorder=4)
        ax.text(hi + 0.10, y, f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]",
                va="center", ha="left", fontsize=8.2, fontweight="bold",
                bbox={"boxstyle": "square,pad=0.12", "facecolor": "white",
                      "edgecolor": "none", "alpha": 1.0}, zorder=6)
    ax.axvline(0, color="#20262C", lw=1.1)
    ax.set_yticks(ypos, [item[0] for item in paired])
    ax.set_xlim(-1.35, 2.75)
    ax.set_xlabel("Feature + KL minus Strong Aug. (pp), paired 95% CI")
    ax.set_title("(b) Paired residual effects and 95% CIs", loc="left", fontweight="bold")
    style(ax, "x")

    # (c) Direct backbone transfer, same pair of methods and five seeds.
    ax = fig.add_subplot(gs[1, 0])
    backbones = ["ResNet-50", "ViT-S/16"]
    strong = [pacs["methods"]["strong_aug"]["macro_mean"],
              vit["methods"]["strong_aug"]["macro_mean"]]
    feature = [pacs["methods"]["feature_plus_kl"]["macro_mean"],
               vit["methods"]["feature_plus_kl"]["macro_mean"]]
    strong_sd = [pacs["methods"]["strong_aug"]["macro_sample_sd"],
                 vit["methods"]["strong_aug"]["macro_sample_sd"]]
    feature_sd = [pacs["methods"]["feature_plus_kl"]["macro_sample_sd"],
                  vit["methods"]["feature_plus_kl"]["macro_sample_sd"]]
    x2 = np.arange(2)
    width2 = 0.32
    bars_s = ax.bar(x2 - width2 / 2, strong, width2, yerr=strong_sd, capsize=4,
                    color=COLORS["strong_aug"], edgecolor="white", linewidth=0.8,
                    label="Strong Aug.")
    bars_f = ax.bar(x2 + width2 / 2, feature, width2, yerr=feature_sd, capsize=4,
                    color=COLORS["feature_plus_kl"], edgecolor="white", linewidth=0.8,
                    label="Feature + KL")
    for bars, values, errors in [(bars_s, strong, strong_sd), (bars_f, feature, feature_sd)]:
        for bar, value, error in zip(bars, values, errors):
            ax.text(bar.get_x() + bar.get_width() / 2, value + error + 0.16,
                    f"{value:.2f}±{error:.2f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#293038")
    for xx, delta in zip(x2, [0.38400742596708426, 0.4384927726256507]):
        ax.text(xx, 84.98, f"Delta {delta:+.2f} pp", ha="center", va="bottom",
                fontsize=8.8, fontweight="bold", color="#4D5660",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                      "edgecolor": "#C9D0D7"})
    ax.set_xticks(x2, backbones)
    ax.set_xlim(-0.48, 1.48)
    ax.set_ylim(84.7, 88.65)
    ax.set_ylabel("PACS macro accuracy (%)")
    ax.set_title("(c) PACS transfer across exact backbones", loc="left", fontweight="bold")
    style(ax)
    ax.legend(loc="upper right", frameon=False)

    # (d) Compact descriptive OAT sensitivity, seed 42 only.
    ax = fig.add_subplot(gs[1, 1])
    sweep_order = ["lambda_f", "lambda_k", "temperature", "augmentation_M"]
    sweep_labels = [r"$\lambda_f$", r"$\lambda_k$", r"$T$", "RandAug. M"]
    sweep_colors = ["#D94733", "#1976B9", "#16835F", "#F0A12B"]
    markers = ["o", "s", "D", "^"]
    offsets = np.linspace(-0.12, 0.12, 4)
    for sweep, label, color, marker, offset in zip(
            sweep_order, sweep_labels, sweep_colors, markers, offsets):
        rows = [row for row in sensitivity if row["sweep"] == sweep]
        target = [float(row["target_macro"]) for row in rows]
        source = [float(row["source_validation_macro"]) for row in rows]
        xx = np.arange(3) + offset
        ax.plot(xx, target, color=color, marker=marker, ms=7.8, lw=2.8,
                markeredgecolor="white", markeredgewidth=0.8, label=label)
        base_index = next(i for i, row in enumerate(rows) if row["reused_default"] == "True")
        ax.scatter(xx[base_index], target[base_index], s=115, facecolor="none",
                   edgecolor=color, linewidth=1.5, zorder=4)
    ax.set_xticks(np.arange(3), ["Low", "Fixed", "High"])
    ax.axvspan(0.73, 1.27, color="#AAB3BC", alpha=0.13, zorder=0)
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(85.8, 88.7)
    ax.set_ylabel("Target macro accuracy (%), seed 42")
    ax.set_title("(d) Descriptive one-factor sensitivity", loc="left", fontweight="bold")
    style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.48, -0.18), ncol=4,
              frameon=False, columnspacing=1.4, handlelength=1.6)
    all_source = [float(row["source_validation_macro"]) for row in sensitivity]
    ax.text(0.99, 0.04,
            f"Open circles: fixed defaults; source-val range "
            f"{min(all_source):.2f}-{max(all_source):.2f}%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.3,
            color="#4D5660",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white",
                  "edgecolor": "#D2D8DE", "alpha": 0.92})

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig2_revision_evidence.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT / "fig2_revision_evidence.pdf")


if __name__ == "__main__":
    main()
