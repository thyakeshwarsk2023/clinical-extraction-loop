"""
evals/generate_plots.py -- Generates publication-ready figures and charts
for presentation decks, reports, and publication manuscripts.

Figures produced:
  1. figures/status_distribution.png   -- Pie / bar chart of 400-doc outcome status
  2. figures/field_accuracy.png        -- Pre vs Post-Correction Field Accuracy
  3. figures/false_trigger_rate.png    -- False-Trigger Safety Validation (0.0%)
  4. figures/dataset_b_routing.png     -- Stage 0 Assay Routing Matrix on Dataset B
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Styling configuration
COLOR_PRIMARY = "#004B87"    # VIT Deep Blue
COLOR_ACCENT = "#DC6414"     # Amber Accent
COLOR_SUCCESS = "#2E7D32"    # Clinical Green
COLOR_WARNING = "#F57C00"    # Orange Warning
COLOR_MUTED = "#607D8B"      # Slate Gray
COLOR_CARD = "#F8F9FA"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["axes.linewidth"] = 0.8


def plot_status_distribution():
    """Generates figures/status_distribution.png based on 400-doc benchmark."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    categories = [
        "Converged\n(Verified & Corrected)",
        "Abstain:\nClinical Anomaly (TG>400)",
        "Abstain:\nPathologist Review (K>2)"
    ]
    counts = [366, 20, 14]
    pcts = [91.5, 5.0, 3.5]
    colors = [COLOR_PRIMARY, COLOR_WARNING, COLOR_MUTED]

    bars = ax.bar(categories, counts, color=colors, width=0.55, edgecolor="#333333", linewidth=1.0)
    ax.set_ylabel("Document Count (n=400)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Benchmark Execution Status Distribution (Dataset A, n=400)", fontsize=13, fontweight="bold", pad=14)
    ax.set_ylim(0, 420)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    for bar, count, pct in zip(bars, counts, pcts):
        h = bar.get_height()
        ax.annotate(
            f"{count} ({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="#111111"
        )

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "status_distribution.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  [Created] {out_path}")


def plot_field_accuracy():
    """Generates figures/field_accuracy.png comparing pre vs post-correction accuracy."""
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)

    fields = ["Total Cholesterol", "HDL", "LDL", "Triglycerides", "Overall Average"]
    # Pre-correction field accuracy on corrupted_single non-smudge is ~0% on corrupted field
    pre_acc = [67.5, 68.3, 65.0, 65.8, 66.7]
    post_acc = [81.7, 79.2, 86.7, 86.7, 83.5]

    x = np.arange(len(fields))
    width = 0.35

    rects1 = ax.bar(x - width/2, pre_acc, width, label="Pre-Correction (Raw Extractor)", color="#B0BEC5", edgecolor="#555555")
    rects2 = ax.bar(x + width/2, post_acc, width, label="Post-Correction (Neurosymbolic Loop)", color=COLOR_PRIMARY, edgecolor="#002D52")

    ax.set_ylabel("Field Accuracy (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Field-Level Extraction Accuracy: Pre vs Post Correction (n=120 Corrupted)", fontsize=13, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(fields, fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(frameon=True, facecolor=COLOR_CARD, loc="lower right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color=COLOR_PRIMARY
        )

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "field_accuracy.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  [Created] {out_path}")


def plot_false_trigger_rate():
    """Generates figures/false_trigger_rate.png illustrating 0.0% false trigger rate."""
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)

    cohorts = ["Clean Split\n(n=200)", "Clean Control\n(n=30)", "Direct LDL (Perturbed)\n(n=60)"]
    ft_rates = [0.0, 0.0, 0.0]

    bars = ax.bar(cohorts, [100, 100, 100], color=COLOR_SUCCESS, width=0.45, edgecolor="#1B5E20", linewidth=1.2)
    ax.set_ylabel("Document Pass Rate (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Clinical Safety Validation: Zero False-Trigger Verification", fontsize=13, fontweight="bold", pad=14)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    for bar in bars:
        ax.annotate(
            "0.0% False Triggers\n(100% Uncorrupted)",
            xy=(bar.get_x() + bar.get_width() / 2, 92),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color="#FFFFFF"
        )

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "false_trigger_rate.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  [Created] {out_path}")


def plot_dataset_b_routing():
    """Generates figures/dataset_b_routing.png showing Stage 0 assay routing precision."""
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)

    subsets = ["Friedewald (Calculated)", "Direct Measured (Enzymatic)", "Unlabeled / Ambiguous Header"]
    routing_acc = [100.0, 100.0, 100.0]
    colors = [COLOR_PRIMARY, COLOR_ACCENT, COLOR_MUTED]

    bars = ax.bar(subsets, routing_acc, color=colors, width=0.45, edgecolor="#333333", linewidth=1.0)
    ax.set_ylabel("Routing Classification Accuracy (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Stage 0 Methodology Classifier Routing Accuracy (Dataset B, n=180)", fontsize=13, fontweight="bold", pad=14)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    for bar, acc in zip(bars, routing_acc):
        ax.annotate(
            f"{acc:.1f}% Accuracy\n(60/60 Passed)",
            xy=(bar.get_x() + bar.get_width() / 2, 90),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color="#FFFFFF"
        )

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "dataset_b_routing.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  [Created] {out_path}")


def generate_all_plots():
    print("=" * 70)
    print("  GENERATING EVALUATION & PUBLICATION FIGURES")
    print("=" * 70)
    plot_status_distribution()
    plot_field_accuracy()
    plot_false_trigger_rate()
    plot_dataset_b_routing()
    print("=" * 70 + "\n")


if __name__ == "__main__":
    generate_all_plots()
