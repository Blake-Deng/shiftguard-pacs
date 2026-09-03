#!/usr/bin/env python3
"""Generate the revised paper figures directly from recorded experiment outputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageEnhance


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "work_latex" / "figures"
DOMAINS = ["Photo", "Art_Painting", "Cartoon", "Sketch"]
DOMAIN_LABELS = ["Photo", "Art painting", "Cartoon", "Sketch"]
SEEDS = [42, 123, 3407]

COLORS = {
    "erm": "#59636F",
    "strong_aug": "#1677B8",
    "feature_plus_kl": "#D84A35",
    "one_way_kl": "#F2A541",
    "adaptive": "#6A4C93",
    "positive": "#167A55",
    "negative": "#C43D3D",
}

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 12.5,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.2,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def old_method_summary(folder: Path, method: str):
    per_domain, per_seed = {}, {seed: [] for seed in SEEDS}
    for domain in DOMAINS:
        values = []
        for seed in SEEDS:
            path = folder / f"{domain}_resnet50_{method}_seed{seed}.json"
            payload = load_json(path)
            value = 100.0 * payload["target_accuracy"]
            values.append(value)
            per_seed[seed].append(value)
        per_domain[domain] = (float(np.mean(values)), float(np.std(values, ddof=1)))
    macros = np.array([np.mean(per_seed[seed]) for seed in SEEDS])
    return per_domain, macros


def corrected_summary():
    records = load_json(ROOT / "runs" / "corrected_ablation" / "summary.json")
    return {record["run_name"]: record for record in records}


def style_axis(ax, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color="#D9DEE3", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def make_results_figure():
    corrected = corrected_summary()
    erm_domains, erm_macros = old_method_summary(ROOT / "runs", "erm")
    methods = ["ERM", "Strong Aug.", "Feature + KL"]
    method_keys = ["erm", "strong_aug", "feature_plus_kl"]
    means = {
        "erm": [erm_domains[d][0] for d in DOMAINS],
        "strong_aug": [corrected["strong_aug"]["per_domain"][d]["mean"] for d in DOMAINS],
        "feature_plus_kl": [corrected["feature_plus_kl"]["per_domain"][d]["mean"] for d in DOMAINS],
    }
    stds = {
        "erm": [erm_domains[d][1] for d in DOMAINS],
        "strong_aug": [corrected["strong_aug"]["per_domain"][d]["std"] for d in DOMAINS],
        "feature_plus_kl": [corrected["feature_plus_kl"]["per_domain"][d]["std"] for d in DOMAINS],
    }

    fig = plt.figure(figsize=(13.8, 8.15), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, left=0.065, right=0.985, bottom=0.075, top=0.955,
                          wspace=0.29, hspace=0.58)

    # (a) Per-domain comparison.
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(4)
    width = 0.245
    for j, (label, key) in enumerate(zip(methods, method_keys)):
        positions = x + (j - 1) * width
        bars = ax.bar(positions, means[key], width, yerr=stds[key], capsize=2.8,
                      color=COLORS[key], edgecolor="white", linewidth=0.65, label=label,
                      error_kw={"elinewidth": 0.9, "capthick": 0.9})
        for bar, value, std in zip(bars, means[key], stds[key]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + std + 0.55, f"{value:.1f}",
                    ha="center", va="bottom", fontsize=8.0, fontweight="bold",
                    color="#25292D",
                    bbox={"boxstyle": "round,pad=0.12", "facecolor": "white",
                          "edgecolor": "none", "alpha": 0.88})
    ax.set_ylim(63, 102.5)
    ax.set_ylabel("Target accuracy (%)")
    ax.set_xticks(x, DOMAIN_LABELS)
    ax.set_title("(a) Accuracy across held-out domains", loc="left", fontweight="bold")
    style_axis(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=3, frameon=False,
              handlelength=1.5, columnspacing=2.0, borderaxespad=0.0)

    # (b) Domain-wise paired descriptive differences.
    ax = fig.add_subplot(gs[0, 1])
    delta = np.array(means["feature_plus_kl"]) - np.array(means["strong_aug"])
    bar_colors = [COLORS["positive"] if value >= 0 else COLORS["negative"] for value in delta]
    bars = ax.barh(np.arange(4), delta, color=bar_colors, height=0.58, edgecolor="white", linewidth=0.7)
    ax.axvline(0, color="#30343B", linewidth=1.0)
    for bar, value in zip(bars, delta):
        if value < -1.0:
            ax.text(value / 2, bar.get_y() + bar.get_height() / 2, f"{value:+.2f} pp",
                    ha="center", va="center", fontweight="bold", fontsize=9.2, color="white")
        else:
            ha = "left" if value >= 0 else "right"
            pad = 0.10 if value >= 0 else -0.10
            ax.text(value + pad, bar.get_y() + bar.get_height() / 2, f"{value:+.2f} pp",
                    ha=ha, va="center", fontweight="bold", fontsize=9.2,
                    color=COLORS["positive"] if value >= 0 else COLORS["negative"])
    ax.set_yticks(np.arange(4), DOMAIN_LABELS)
    ax.tick_params(axis="y", pad=7)
    ax.invert_yaxis()
    ax.set_xlim(-2.65, 2.25)
    ax.set_xlabel("Feature + KL minus Strong Augmentation")
    ax.set_title("(b) Consistency helps the stylized domains", loc="left", fontweight="bold")
    style_axis(ax, "x")
    ax.text(0.985, 0.04, "Macro difference: +0.26 pp", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9.4, fontweight="bold", color="#30343B",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#CCD3DA"})

    # (c) Corrected ablation relative to the direct augmentation baseline.
    ax = fig.add_subplot(gs[1, 0])
    variants = ["strong_aug", "one_way_kl", "feature_plus_kl", "adaptive"]
    variant_labels = ["Strong Aug.", "+ one-way KL", "+ feature + KL", "+ adaptive gate"]
    matrix = np.array([[corrected[v]["per_domain"][d]["mean"] for d in DOMAINS] +
                       [corrected[v]["macro_mean"]] for v in variants])
    relative = matrix - matrix[0:1, :]
    cmap = LinearSegmentedColormap.from_list("div", ["#2B6CB0", "#F7F7F7", "#C53030"])
    norm = TwoSlopeNorm(vmin=-2.5, vcenter=0.0, vmax=2.0)
    image = ax.imshow(relative, cmap=cmap, norm=norm, aspect="auto")
    for row in range(relative.shape[0]):
        for col in range(relative.shape[1]):
            value = relative[row, col]
            color = "white" if abs(value) > 1.25 else "#20252A"
            label = "baseline" if row == 0 else f"{value:+.2f}"
            ax.text(col, row, label, ha="center", va="center", fontsize=9.0,
                    fontweight="bold" if row == 2 else "normal", color=color)
    ax.set_xticks(np.arange(5), DOMAIN_LABELS + ["Macro"])
    ax.set_yticks(np.arange(4), variant_labels)
    ax.set_title("(c) Corrected ablation (difference from Strong Aug., pp)", loc="left", fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Accuracy change (pp)", fontsize=9.2)
    cbar.ax.tick_params(labelsize=8.5)

    # (d) Seed-level macro results: the correct unit for macro variability.
    ax = fig.add_subplot(gs[1, 1])
    macro_by_variant = {
        "strong_aug": np.array(corrected["strong_aug"]["macro_by_seed"]),
        "one_way_kl": np.array(corrected["one_way_kl"]["macro_by_seed"]),
        "feature_plus_kl": np.array(corrected["feature_plus_kl"]["macro_by_seed"]),
        "adaptive": np.array(corrected["adaptive"]["macro_by_seed"]),
    }
    xpos = np.arange(4)
    seed_matrix = np.column_stack([macro_by_variant[v] for v in variants])
    seed_colors = ["#0072B2", "#D55E00", "#009E73"]
    seed_markers = ["o", "s", "D"]
    lower, upper = seed_matrix.min(axis=0), seed_matrix.max(axis=0)
    means_d = seed_matrix.mean(axis=0)
    stds_d = seed_matrix.std(axis=0, ddof=1)
    baseline_mean = means_d[0]
    baseline_std = stds_d[0]
    ax.axhspan(baseline_mean - baseline_std, baseline_mean + baseline_std,
               color=COLORS["strong_aug"], alpha=0.10, zorder=0)
    ax.fill_between(xpos, lower, upper, color="#7F8C99", alpha=0.12, zorder=1)
    for seed_idx, (color, marker) in enumerate(zip(seed_colors, seed_markers)):
        ax.plot(xpos, seed_matrix[seed_idx], color=color, linewidth=2.0, alpha=0.85,
                marker=marker, markersize=6.5, markeredgecolor="white",
                markeredgewidth=0.75, label=f"seed {SEEDS[seed_idx]}", zorder=3)
    ax.plot(xpos, means_d, color="#20252A", linewidth=2.8, alpha=0.82, zorder=4)
    for i, variant in enumerate(variants):
        ax.errorbar(i, means_d[i], yerr=stds_d[i], fmt="o", markersize=9.5,
                    markerfacecolor="white", markeredgecolor=COLORS[variant],
                    markeredgewidth=2.1, ecolor=COLORS[variant], elinewidth=1.5,
                    capsize=4, zorder=5)
        label_y = max(upper[i], means_d[i] + stds_d[i]) + 0.18
        ax.text(i, label_y, f"{means_d[i]:.2f}±{stds_d[i]:.2f}",
                ha="center", va="bottom", fontsize=8.7, fontweight="bold",
                color="#25292D",
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white",
                      "edgecolor": "none", "alpha": 0.90})
    ax.annotate("+0.26 pp mean", xy=(2, means_d[2]), xytext=(1.48, 85.45),
                ha="center", fontsize=8.6, fontweight="bold", color=COLORS["feature_plus_kl"],
                arrowprops={"arrowstyle": "->", "color": COLORS["feature_plus_kl"], "lw": 1.1})
    ax.set_xticks(xpos, ["Strong\nAug.", "One-way\nKL", "Feature\n+ KL", "Adaptive\ngate"])
    ax.set_ylabel("Four-domain macro accuracy (%)")
    ax.set_xlim(-0.28, 3.28)
    ax.set_ylim(85.05, 89.0)
    ax.set_title("(d) Paired macro trajectories across seeds", loc="left", fontweight="bold")
    style_axis(ax)
    ax.legend(loc="lower left", ncol=3, frameon=False, handlelength=1.5,
              columnspacing=1.0, fontsize=8.5)

    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = OUT / f"fig2_corrected_results.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_robustness():
    rows = []
    with (ROOT / "runs" / "corrected_robustness" / "results.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "seed": int(row["seed"]), "method": row["method"],
                "corruption": row["corruption"], "severity": int(row["severity"]),
                "accuracy": 100.0 * float(row["accuracy"]),
            })
    return rows


def make_robustness_figure():
    rows = load_robustness()
    kinds = ["gaussian_noise", "blur", "brightness", "contrast", "jpeg"]
    labels = ["Noise", "Blur", "Brightness", "Contrast", "JPEG"]
    kind_colors = ["#E45756", "#4C78A8", "#F2A541", "#2A9D8F", "#7A5195"]
    fig, ax = plt.subplots(figsize=(7.15, 3.35), constrained_layout=True)
    centers = np.arange(len(kinds)) * 1.45
    offsets = {"strong_aug": -0.23, "feature_plus_kl": 0.23}
    markers = {"strong_aug": "o", "feature_plus_kl": "D"}
    names = {"strong_aug": "Strong Aug.", "feature_plus_kl": "Feature + KL"}
    deltas = []
    for center, kind, color in zip(centers, kinds, kind_colors):
        seed_values = {}
        for method in offsets:
            values = []
            for seed in SEEDS:
                scores = [r["accuracy"] for r in rows if r["method"] == method and
                          r["corruption"] == kind and r["seed"] == seed]
                values.append(np.mean(scores))
            seed_values[method] = np.array(values)
        for seed_idx in range(3):
            ax.plot([center + offsets["strong_aug"], center + offsets["feature_plus_kl"]],
                    [seed_values["strong_aug"][seed_idx], seed_values["feature_plus_kl"][seed_idx]],
                    color=color, linewidth=1.05, alpha=0.45, zorder=1)
        for method in offsets:
            vals = seed_values[method]
            x = center + offsets[method]
            ax.scatter(np.full(3, x), vals, s=34, marker=markers[method], color=color,
                       edgecolor="white", linewidth=0.65, alpha=0.82, zorder=3)
            mean, std = vals.mean(), vals.std(ddof=1)
            ax.errorbar(x, mean, yerr=std, fmt=markers[method], markersize=7.2,
                        markerfacecolor="white", markeredgecolor="#20252A", markeredgewidth=1.3,
                        ecolor="#20252A", elinewidth=1.15, capsize=3.0, zorder=4)
        delta = seed_values["feature_plus_kl"].mean() - seed_values["strong_aug"].mean()
        deltas.append(delta)
        top = max(seed_values["strong_aug"].max(), seed_values["feature_plus_kl"].max())
        ax.text(center, min(88.1, top + 0.75), f"Δ {delta:+.2f}", ha="center", va="bottom",
                fontsize=8.9, fontweight="bold", color=color)
    ax.set_xticks(centers, labels)
    ax.set_ylabel("Sketch accuracy averaged over severities (%)")
    ax.set_ylim(78.0, 89.2)
    ax.set_title("Three-seed corruption stress test", loc="left", fontweight="bold")
    style_axis(ax)
    ax.scatter([], [], marker="o", s=42, color="#5A626A", label="Strong Aug.")
    ax.scatter([], [], marker="D", s=42, color="#5A626A", label="Feature + KL")
    ax.legend(loc="lower left", ncol=2, frameon=False, handletextpad=0.45, columnspacing=1.4)
    ax.text(0.99, 0.045, "Thin links pair identical training seeds; error bars: mean ± seed SD",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.3, color="#4C535A")
    for boundary in (centers[:-1] + centers[1:]) / 2:
        ax.axvline(boundary, color="#E7EAED", linewidth=0.8, zorder=0)
    for extension in ("pdf", "png"):
        fig.savefig(OUT / f"fig3_corrected_robustness.{extension}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def crop_square(image: Image.Image, size=500):
    image = image.convert("RGB")
    w, h = image.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return image.crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.LANCZOS)


def draw_box(ax, xy, width, height, text, face, edge="#263238", fontsize=10.5):
    patch = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.012,rounding_size=0.02",
                           linewidth=1.15, edgecolor=edge, facecolor=face)
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="#20252A")
    return patch


def arrow(ax, start, end, color="#4C5661", style="-|>", linewidth=1.4):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=12,
                                linewidth=linewidth, color=color, connectionstyle="arc3,rad=0"))


def make_overview_figure():
    source_paths = [
        ROOT / "data/PACS/photo/dog/04392.jpg",
        ROOT / "data/PACS/cartoon/dog/02048.jpg",
        ROOT / "data/PACS/sketch/dog/06062.jpg",
    ]
    target_path = ROOT / "data/PACS/art_painting/dog/00000.jpg"
    source_images = [crop_square(Image.open(path)) for path in source_paths]
    target_image = crop_square(Image.open(target_path))
    fig, ax = plt.subplots(figsize=(13.8, 3.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.105, 0.94, "Source domains only", ha="center", va="center", fontsize=12.2,
            fontweight="bold", color="#17324D")
    image_x = [0.025, 0.095, 0.165]
    labels = ["Photo", "Cartoon", "Sketch"]
    for x, image, label in zip(image_x, source_images, labels):
        ax.imshow(image, extent=(x, x + 0.058, 0.58, 0.84), aspect="auto", zorder=2)
        ax.add_patch(plt.Rectangle((x, 0.58), 0.058, 0.26, fill=False, edgecolor="#355C7D", linewidth=1.3))
        ax.text(x + 0.029, 0.54, label, ha="center", va="top", fontsize=8.8)
    ax.text(0.105, 0.44, "same PACS class: dog", ha="center", fontsize=9.0, color="#4C5661")

    arrow(ax, (0.225, 0.70), (0.276, 0.70))
    draw_box(ax, (0.278, 0.63), 0.088, 0.14, "weak view", "#DCECF7")
    draw_box(ax, (0.278, 0.36), 0.088, 0.14, "strong view", "#FBE0D6")
    arrow(ax, (0.245, 0.69), (0.278, 0.43))

    draw_box(ax, (0.414, 0.48), 0.105, 0.17, "shared\nencoder", "#E9EEF2", fontsize=11.0)
    arrow(ax, (0.366, 0.70), (0.414, 0.59))
    arrow(ax, (0.366, 0.43), (0.414, 0.54))
    draw_box(ax, (0.568, 0.48), 0.096, 0.17, "classifier", "#E8E1F4", fontsize=11.0)
    arrow(ax, (0.519, 0.565), (0.568, 0.565))

    draw_box(ax, (0.705, 0.67), 0.125, 0.13, "two-view CE", "#E2F0E8")
    draw_box(ax, (0.705, 0.46), 0.125, 0.13, "feature cosine", "#FFF0C9")
    draw_box(ax, (0.705, 0.25), 0.125, 0.13, "one-way KL", "#F9D8D4")
    arrow(ax, (0.664, 0.565), (0.705, 0.735))
    arrow(ax, (0.519, 0.53), (0.705, 0.525))
    arrow(ax, (0.664, 0.535), (0.705, 0.315))
    ax.text(0.767, 0.18, "detached weak teacher", ha="center", fontsize=8.8, color="#8C2D2D")

    draw_box(ax, (0.867, 0.43), 0.082, 0.20, "weighted\nsum", "#D7EBD8", fontsize=11.0)
    for y in (0.735, 0.525, 0.315):
        arrow(ax, (0.830, y), (0.867, 0.53))

    ax.imshow(target_image, extent=(0.955, 0.995, 0.71, 0.90), aspect="auto", alpha=0.95)
    ax.add_patch(plt.Rectangle((0.955, 0.71), 0.04, 0.19, fill=False, edgecolor="#B83A3A", linewidth=1.3))
    ax.text(0.975, 0.67, "held-out\nArt painting", ha="center", va="top", fontsize=8.2, color="#9D2525")
    ax.plot([0.948, 0.948], [0.12, 0.93], linestyle="--", color="#B83A3A", linewidth=1.2)
    ax.text(0.948, 0.08, "evaluated once after checkpoint selection", ha="right", va="center",
            fontsize=8.5, color="#9D2525")

    for extension in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_protocol.{extension}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    make_results_figure()
    make_robustness_figure()
    make_overview_figure()
    print(f"Wrote revised figures to {OUT}")
