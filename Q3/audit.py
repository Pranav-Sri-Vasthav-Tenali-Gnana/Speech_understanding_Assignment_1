import argparse
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SPEAKERS_TXT = "../Q2/data/LibriSpeech/LibriSpeech/SPEAKERS.TXT"
TARGET_SUBSETS = {"train-clean-100", "train-clean-360", "test-clean"}


def parse_speakers(speakers_txt):
    speakers = []
    with open(speakers_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 5:
                continue
            try:
                sid = int(parts[0])
            except ValueError:
                continue
            gender = parts[1].upper()
            subset = parts[2]
            try:
                minutes = float(parts[3])
            except ValueError:
                continue
            name = parts[4]
            speakers.append({
                "id": sid,
                "gender": gender,
                "subset": subset,
                "minutes": minutes,
                "name": name,
            })
    return speakers


def compute_audit_stats(speakers, subsets):
    filtered = [s for s in speakers if s["subset"] in subsets]

    total = len(filtered)
    gender_counts = defaultdict(int)
    gender_minutes = defaultdict(float)
    subset_gender_counts = defaultdict(lambda: defaultdict(int))
    subset_gender_minutes = defaultdict(lambda: defaultdict(float))

    for s in filtered:
        g = s["gender"]
        sub = s["subset"]
        gender_counts[g] += 1
        gender_minutes[g] += s["minutes"]
        subset_gender_counts[sub][g] += 1
        subset_gender_minutes[sub][g] += s["minutes"]

    stats = {
        "total_speakers": total,
        "gender_counts": dict(gender_counts),
        "gender_minutes": dict(gender_minutes),
        "subset_breakdown": {
            sub: {
                "counts": dict(subset_gender_counts[sub]),
                "minutes": dict(subset_gender_minutes[sub]),
            }
            for sub in sorted(subsets)
        },
    }

    male = gender_counts.get("M", 0)
    female = gender_counts.get("F", 0)
    if total > 0:
        stats["gender_ratio_M"] = round(male / total, 4)
        stats["gender_ratio_F"] = round(female / total, 4)
        stats["imbalance_ratio"] = round(max(male, female) / max(min(male, female), 1), 4)

    total_min = gender_minutes.get("M", 0) + gender_minutes.get("F", 0)
    if total_min > 0:
        stats["speaking_time_ratio_M"] = round(gender_minutes.get("M", 0) / total_min, 4)
        stats["speaking_time_ratio_F"] = round(gender_minutes.get("F", 0) / total_min, 4)

    return stats


def documentation_debt_report(speakers, subsets):
    all_subsets_in_file = set(s["subset"] for s in speakers)
    missing_metadata = []

    filtered = [s for s in speakers if s["subset"] in subsets]
    missing_name = [s for s in filtered if not s["name"]]
    unknown_gender = [s for s in filtered if s["gender"] not in ("M", "F")]
    zero_minutes = [s for s in filtered if s["minutes"] == 0.0]

    debt = {
        "missing_name_field": len(missing_name),
        "unknown_gender": len(unknown_gender),
        "zero_duration_entries": len(zero_minutes),
        "undocumented_age": "all speakers — no age metadata in LibriSpeech",
        "undocumented_dialect": "all speakers — no dialect/accent metadata in LibriSpeech",
        "undocumented_recording_conditions": "no SNR/room/device metadata",
    }
    return debt


def plot_gender_distribution(stats, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = stats["gender_counts"]
    labels = ["Male", "Female"]
    values = [counts.get("M", 0), counts.get("F", 0)]
    colors = ["#4C72B0", "#DD8452"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].bar(labels, values, color=colors, width=0.5)
    axes[0].set_title("Speaker Count by Gender")
    axes[0].set_ylabel("Number of Speakers")
    for i, v in enumerate(values):
        axes[0].text(i, v + 0.5, str(v), ha="center", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    minutes = stats["gender_minutes"]
    min_vals = [minutes.get("M", 0), minutes.get("F", 0)]
    axes[1].bar(labels, min_vals, color=colors, width=0.5)
    axes[1].set_title("Total Speaking Time by Gender (minutes)")
    axes[1].set_ylabel("Minutes")
    for i, v in enumerate(min_vals):
        axes[1].text(i, v + 1, f"{v:.1f}", ha="center", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].pie(
        values,
        labels=[f"{l}\n({v})" for l, v in zip(labels, values)],
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[2].set_title("Gender Share (Speakers)")

    plt.suptitle("LibriSpeech Gender Representation Audit", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "gender_distribution.png", dpi=150)
    plt.close()


def plot_subset_breakdown(stats, out_dir):
    out_dir = Path(out_dir)
    subsets = sorted(stats["subset_breakdown"].keys())
    male_counts = [stats["subset_breakdown"][s]["counts"].get("M", 0) for s in subsets]
    female_counts = [stats["subset_breakdown"][s]["counts"].get("F", 0) for s in subsets]
    male_min = [stats["subset_breakdown"][s]["minutes"].get("M", 0) for s in subsets]
    female_min = [stats["subset_breakdown"][s]["minutes"].get("F", 0) for s in subsets]

    x = np.arange(len(subsets))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(x - width / 2, male_counts, width, label="Male", color="#4C72B0")
    axes[0].bar(x + width / 2, female_counts, width, label="Female", color="#DD8452")
    axes[0].set_title("Speakers per Subset by Gender")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(subsets, rotation=15)
    axes[0].set_ylabel("Number of Speakers")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x - width / 2, male_min, width, label="Male", color="#4C72B0")
    axes[1].bar(x + width / 2, female_min, width, label="Female", color="#DD8452")
    axes[1].set_title("Speaking Time per Subset by Gender (minutes)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(subsets, rotation=15)
    axes[1].set_ylabel("Minutes")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("Per-Subset Gender Breakdown", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "subset_breakdown.png", dpi=150)
    plt.close()


def plot_imbalance_heatmap(stats, out_dir):
    out_dir = Path(out_dir)
    subsets = sorted(stats["subset_breakdown"].keys())

    ratios = []
    for sub in subsets:
        m = stats["subset_breakdown"][sub]["counts"].get("M", 0)
        f = stats["subset_breakdown"][sub]["counts"].get("F", 0)
        total = m + f
        ratios.append([m / total if total else 0, f / total if total else 0])

    ratios = np.array(ratios)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(ratios.T, cmap="RdYlGn", vmin=0.3, vmax=0.7, aspect="auto")
    ax.set_xticks(range(len(subsets)))
    ax.set_xticklabels(subsets, rotation=15)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Male", "Female"])
    ax.set_title("Gender Ratio Heatmap per Subset\n(green=balanced, red=skewed)")

    for i in range(len(subsets)):
        for j in range(2):
            ax.text(i, j, f"{ratios[i, j]:.2f}", ha="center", va="center", fontsize=11, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Proportion")
    plt.tight_layout()
    plt.savefig(out_dir / "imbalance_heatmap.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speakers_txt", default=SPEAKERS_TXT)
    parser.add_argument("--subsets", nargs="+", default=list(TARGET_SUBSETS))
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()

    plots_dir = Path(args.out_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Parsing SPEAKERS.TXT...")
    speakers = parse_speakers(args.speakers_txt)
    print(f"Total entries in file: {len(speakers)}")

    subsets = set(args.subsets)
    stats = compute_audit_stats(speakers, subsets)
    debt = documentation_debt_report(speakers, subsets)

    print("\n=== Gender Representation Audit ===")
    print(f"Total speakers (selected subsets): {stats['total_speakers']}")
    print(f"Male:   {stats['gender_counts'].get('M', 0)} speakers  "
          f"({stats.get('gender_ratio_M', 0)*100:.1f}%)")
    print(f"Female: {stats['gender_counts'].get('F', 0)} speakers  "
          f"({stats.get('gender_ratio_F', 0)*100:.1f}%)")
    print(f"Imbalance ratio (majority/minority): {stats.get('imbalance_ratio', 1.0):.2f}x")
    print(f"\nSpeaking time — Male: {stats['gender_minutes'].get('M', 0):.1f} min  "
          f"Female: {stats['gender_minutes'].get('F', 0):.1f} min")

    print("\n=== Per-Subset Breakdown ===")
    for sub, info in stats["subset_breakdown"].items():
        m = info["counts"].get("M", 0)
        f = info["counts"].get("F", 0)
        print(f"  {sub}: M={m}, F={f}, total={m+f}")

    print("\n=== Documentation Debt ===")
    for k, v in debt.items():
        print(f"  {k}: {v}")

    audit_report = {"stats": stats, "documentation_debt": debt}
    with open(Path(args.out_dir) / "audit_report.json", "w") as fp:
        json.dump(audit_report, fp, indent=2)

    print("\nGenerating plots...")
    plot_gender_distribution(stats, plots_dir)
    plot_subset_breakdown(stats, plots_dir)
    plot_imbalance_heatmap(stats, plots_dir)
    print(f"Plots saved to {plots_dir}")
    print(f"Audit report saved to {args.out_dir}/audit_report.json")


if __name__ == "__main__":
    main()
