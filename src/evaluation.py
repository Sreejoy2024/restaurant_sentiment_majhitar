"""
Evaluation & Visualization Module
Generates metrics, confusion matrices, word clouds, and performance plots.
Majhitar, Sikkim Restaurant Reviews - NLP Assignment
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List
from collections import Counter

warnings.filterwarnings("ignore")

# ── Style ──
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

PALETTE = {
    "positive": "#2ecc71",
    "neutral": "#f39c12",
    "negative": "#e74c3c",
    "primary": "#2c3e50",
    "secondary": "#3498db",
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ─────────────────────────────────────────────
# 1. CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, model_name: str,
                           labels: List[str], save_dir: str = "evaluation"):
    ensure_dir(save_dir)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", labelpad=10)
    ax.set_ylabel("True Label", labelpad=10)
    plt.tight_layout()
    fname = f"{save_dir}/confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Saved: {fname}")
    return fname


# ─────────────────────────────────────────────
# 2. MODEL COMPARISON BAR CHART
# ─────────────────────────────────────────────

def plot_model_comparison(results: Dict, save_dir: str = "evaluation"):
    ensure_dir(save_dir)

    model_names = list(results.keys())
    accuracies = [results[m]["accuracy"] for m in model_names]
    f1_scores = [results[m]["f1_weighted"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy",
                   color="#3498db", edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, f1_scores, width, label="F1 (Weighted)",
                   color="#e67e22", edgecolor="white", linewidth=0.8)

    # Value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Sentiment Model Performance Comparison\nMajhitar Restaurant Reviews",
                 fontweight="bold", pad=15)
    ax.set_ylabel("Score", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim(0, 1.12)
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fname = f"{save_dir}/model_comparison.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Saved: {fname}")
    return fname


# ─────────────────────────────────────────────
# 3. SENTIMENT DISTRIBUTION
# ─────────────────────────────────────────────

def plot_sentiment_distribution(df: pd.DataFrame, save_dir: str = "evaluation"):
    ensure_dir(save_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Overall distribution ──
    counts = df["sentiment_label"].value_counts()
    colors = [PALETTE.get(l, "#95a5a6") for l in counts.index]
    wedges, texts, autotexts = axes[0].pie(
        counts.values, labels=counts.index, colors=colors,
        autopct="%1.1f%%", startangle=140, pctdistance=0.85,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    axes[0].set_title("Overall Sentiment Distribution", fontweight="bold", pad=15)

    # ── Per-restaurant stacked bar ──
    restaurant_sentiment = df.groupby(["restaurant_name", "sentiment_label"]).size().unstack(fill_value=0)
    # Normalize
    restaurant_sentiment_pct = restaurant_sentiment.div(restaurant_sentiment.sum(axis=1), axis=0) * 100

    # Shorten restaurant names
    restaurant_sentiment_pct.index = [n[:20] + "..." if len(n) > 20 else n
                                        for n in restaurant_sentiment_pct.index]

    bar_colors = [PALETTE.get(c, "#95a5a6") for c in restaurant_sentiment_pct.columns]
    restaurant_sentiment_pct.plot(
        kind="bar", stacked=True, ax=axes[1],
        color=bar_colors, edgecolor="white", linewidth=0.5
    )
    axes[1].set_title("Sentiment by Restaurant (%)", fontweight="bold", pad=15)
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(axis="y", alpha=0.3, linestyle="--")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    fname = f"{save_dir}/sentiment_distribution.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Saved: {fname}")
    return fname


# ─────────────────────────────────────────────
# 4. RATING VS SENTIMENT SCATTER
# ─────────────────────────────────────────────

def plot_rating_vs_sentiment(df: pd.DataFrame, save_dir: str = "evaluation"):
    ensure_dir(save_dir)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Add small jitter for readability
    jitter = np.random.uniform(-0.1, 0.1, len(df))
    colors = [PALETTE.get(l, "#95a5a6") for l in df["sentiment_label"]]

    ax.scatter(df["rating"] + jitter, df["sentiment_score"], c=colors,
               alpha=0.5, s=30, edgecolors="none")

    # Legend
    patches = [mpatches.Patch(color=PALETTE[k], label=k.capitalize())
               for k in ["positive", "neutral", "negative"]]
    ax.legend(handles=patches, loc="upper left", framealpha=0.9)

    ax.set_title("Star Rating vs Sentiment Score\nMajhitar Restaurant Reviews",
                 fontweight="bold", pad=15)
    ax.set_xlabel("Star Rating (1–5)", labelpad=10)
    ax.set_ylabel("Sentiment Score", labelpad=10)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fname = f"{save_dir}/rating_vs_sentiment.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Saved: {fname}")
    return fname


# ─────────────────────────────────────────────
# 5. TOP FREQUENT WORDS
# ─────────────────────────────────────────────

def plot_top_words(df: pd.DataFrame, save_dir: str = "evaluation", top_n: int = 20):
    ensure_dir(save_dir)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, sentiment in zip(axes, ["positive", "neutral", "negative"]):
        subset = df[df["sentiment_label"] == sentiment]["filtered_text"].dropna()
        words = " ".join(subset).split()
        freq = Counter(words).most_common(top_n)
        if not freq:
            ax.set_visible(False)
            continue
        words_list, counts = zip(*freq)
        color = PALETTE[sentiment]
        bars = ax.barh(list(words_list)[::-1], list(counts)[::-1],
                       color=color, edgecolor="white", linewidth=0.5)
        ax.set_title(f"Top Words — {sentiment.capitalize()} Reviews",
                     fontweight="bold", color=color, pad=10)
        ax.set_xlabel("Frequency")
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fname = f"{save_dir}/top_words_by_sentiment.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Saved: {fname}")
    return fname


# ─────────────────────────────────────────────
# 6. RESTAURANT RATING COMPARISON
# ─────────────────────────────────────────────

def plot_restaurant_ratings(df: pd.DataFrame, save_dir: str = "evaluation"):
    ensure_dir(save_dir)
    fig, ax = plt.subplots(figsize=(12, 6))

    avg_ratings = df.groupby("restaurant_name")["rating"].mean().sort_values(ascending=False)
    pos_pct = df.groupby("restaurant_name").apply(
        lambda x: (x["sentiment_label"] == "positive").mean() * 100
    ).reindex(avg_ratings.index)

    x = np.arange(len(avg_ratings))
    # Shorten names
    short_names = [n[:18] + ".." if len(n) > 18 else n for n in avg_ratings.index]

    bars = ax.bar(x, avg_ratings.values, color=[
        plt.cm.RdYlGn(v / 5.0) for v in avg_ratings.values
    ], edgecolor="white", linewidth=0.8)

    # Overlay positive % as line
    ax2 = ax.twinx()
    ax2.plot(x, pos_pct.values, color="#3498db", marker="o",
             linewidth=2, markersize=7, label="Positive %")
    ax2.set_ylabel("Positive Review %", color="#3498db", labelpad=10)
    ax2.tick_params(axis="y", labelcolor="#3498db")
    ax2.set_ylim(0, 110)

    for bar, val in zip(bars, avg_ratings.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{val:.1f}★", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Average Rating & Positive Review % by Restaurant\nMajhitar, Sikkim",
                 fontweight="bold", pad=15)
    ax.set_ylabel("Average Star Rating", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=25, ha="right")
    ax.set_ylim(0, 6)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    ax2.legend(loc="upper right")
    plt.tight_layout()
    fname = f"{save_dir}/restaurant_ratings.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Saved: {fname}")
    return fname


# ─────────────────────────────────────────────
# 7. CROSS-VALIDATION SCORES
# ─────────────────────────────────────────────

def plot_cv_scores(results: Dict, save_dir: str = "evaluation"):
    ensure_dir(save_dir)
    ml_results = {k: v for k, v in results.items() if "cv_mean" in v}
    if not ml_results:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    models = list(ml_results.keys())
    means = [ml_results[m]["cv_mean"] for m in models]
    stds = [ml_results[m]["cv_std"] for m in models]

    bars = ax.bar(models, means, yerr=stds, capsize=8,
                  color=["#3498db", "#e67e22", "#9b59b6"],
                  edgecolor="white", linewidth=0.8, error_kw={"linewidth": 2})

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds) + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("5-Fold Cross-Validation F1 Scores\n(Mean ± Std)", fontweight="bold", pad=15)
    ax.set_ylabel("F1 Score (Weighted)")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fname = f"{save_dir}/cross_validation_scores.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Saved: {fname}")
    return fname


# ─────────────────────────────────────────────
# 8. SAVE EVALUATION REPORT
# ─────────────────────────────────────────────

def save_evaluation_report(results: Dict, save_dir: str = "evaluation"):
    ensure_dir(save_dir)
    lines = []
    lines.append("=" * 70)
    lines.append("  SENTIMENT ANALYSIS EVALUATION REPORT")
    lines.append("  Majhitar/Majitar, Sikkim — Restaurant Reviews")
    lines.append("=" * 70)

    for model_name, res in results.items():
        lines.append(f"\n{'─'*60}")
        lines.append(f"  MODEL: {model_name}")
        lines.append(f"{'─'*60}")
        lines.append(f"  Accuracy:    {res['accuracy']:.4f}")
        lines.append(f"  F1 (Weighted): {res['f1_weighted']:.4f}")
        if "cv_mean" in res:
            lines.append(f"  CV F1 Mean:  {res['cv_mean']:.4f} ± {res['cv_std']:.4f}")
        lines.append("\n  Classification Report:")
        lines.append(res["report"])

    lines.append("\n" + "=" * 70)

    report_path = f"{save_dir}/evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[✓] Evaluation report saved to {report_path}")
    return report_path


def run_all_visualizations(df: pd.DataFrame, results: Dict, save_dir: str = "evaluation"):
    """Generate all evaluation plots."""
    print("\n[→] Generating evaluation visualizations...")

    labels = ["positive", "neutral", "negative"]

    # Confusion matrices
    for model_name, res in results.items():
        if "confusion_matrix" in res:
            plot_confusion_matrix(res["confusion_matrix"], model_name, labels, save_dir)

    plot_model_comparison(results, save_dir)
    plot_sentiment_distribution(df, save_dir)
    plot_rating_vs_sentiment(df, save_dir)
    plot_top_words(df, save_dir)
    plot_restaurant_ratings(df, save_dir)
    plot_cv_scores(results, save_dir)
    save_evaluation_report(results, save_dir)

    print("[✓] All visualizations saved to", save_dir)
