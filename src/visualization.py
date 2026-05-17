"""
Visualization Module
====================
Generate charts for benchmark results analysis.
Produces plots for token reduction, latency, similarity scores,
and summary comparison charts.
"""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt


# Consistent method ordering
_METHOD_ORDER = ["full", "extractive", "mmr", "summary", "hybrid"]

# Color palette
_COLORS = {
    "blue": "#4A90D9",
    "red": "#E74C3C",
    "retrieval": "#3498DB",
    "compression": "#E67E22",
    "generation": "#2ECC71",
    "purple": "#9B59B6",
    "teal": "#1ABC9C",
    "dark": "#2C3E50",
}


def load_metrics(csv_path: str) -> pd.DataFrame:
    """
    Load benchmark metrics from CSV file.

    Args:
        csv_path: Path to metrics.csv.

    Returns:
        DataFrame with benchmark results.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    return pd.read_csv(path)


def _ordered_reindex(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex DataFrame by standard method order, dropping missing."""
    available = [m for m in _METHOD_ORDER if m in df.index]
    return df.reindex(available)


def plot_token_reduction(df: pd.DataFrame, output_dir: str) -> str:
    """
    Plot average token reduction by compression method.

    Args:
        df: DataFrame with benchmark metrics.
        output_dir: Directory to save the plot.

    Returns:
        Path to saved plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    grouped = df.groupby("method").agg({
        "tokens_before": "mean",
        "tokens_after": "mean",
        "reduction_percent": "mean",
    })
    grouped = _ordered_reindex(grouped)

    if grouped.empty:
        plt.close(fig)
        return ""

    methods = grouped.index.tolist()
    x = range(len(methods))
    width = 0.35

    ax.bar(
        [i - width/2 for i in x],
        grouped["tokens_before"],
        width,
        label="Before Compression",
        color=_COLORS["blue"],
        alpha=0.85,
    )
    ax.bar(
        [i + width/2 for i in x],
        grouped["tokens_after"],
        width,
        label="After Compression",
        color=_COLORS["red"],
        alpha=0.85,
    )

    # Add reduction % labels on top
    for i, method in enumerate(methods):
        reduction = grouped.loc[method, "reduction_percent"]
        y_pos = max(grouped.loc[method, "tokens_before"],
                    grouped.loc[method, "tokens_after"]) + 20
        ax.annotate(
            f"{reduction:.1f}%",
            xy=(i, y_pos),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color=_COLORS["dark"],
        )

    ax.set_xlabel("Compression Method", fontsize=12)
    ax.set_ylabel("Average Token Count", fontsize=12)
    ax.set_title("Token Reduction by Compression Method", fontsize=14, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels([m.capitalize() for m in methods])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / "token_reduction.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved: {output_path}")
    return str(output_path)


def plot_latency(df: pd.DataFrame, output_dir: str) -> str:
    """
    Plot average latency breakdown by compression method.

    Args:
        df: DataFrame with benchmark metrics.
        output_dir: Directory to save the plot.

    Returns:
        Path to saved plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    grouped = df.groupby("method").agg({
        "latency_retrieval": "mean",
        "latency_compression": "mean",
        "latency_generation": "mean",
        "latency_total": "mean",
    })
    grouped = _ordered_reindex(grouped)

    if grouped.empty:
        plt.close(fig)
        return ""

    methods = grouped.index.tolist()
    x = range(len(methods))
    width = 0.25

    ax.bar(
        [i - width for i in x],
        grouped["latency_retrieval"],
        width,
        label="Retrieval",
        color=_COLORS["retrieval"],
        alpha=0.85,
    )
    ax.bar(
        list(x),
        grouped["latency_compression"],
        width,
        label="Compression",
        color=_COLORS["compression"],
        alpha=0.85,
    )
    ax.bar(
        [i + width for i in x],
        grouped["latency_generation"],
        width,
        label="Generation",
        color=_COLORS["generation"],
        alpha=0.85,
    )

    # Total latency line
    ax.plot(
        list(x),
        grouped["latency_total"],
        "ko-",
        label="Total",
        markersize=8,
        linewidth=2,
    )

    ax.set_xlabel("Compression Method", fontsize=12)
    ax.set_ylabel("Average Latency (seconds)", fontsize=12)
    ax.set_title("Pipeline Latency by Compression Method", fontsize=14, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels([m.capitalize() for m in methods])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / "latency.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved: {output_path}")
    return str(output_path)


def plot_similarity(df: pd.DataFrame, output_dir: str) -> str:
    """
    Plot semantic similarity scores by compression method.

    Args:
        df: DataFrame with benchmark metrics.
        output_dir: Directory to save the plot.

    Returns:
        Path to saved plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter out 'full' method since similarity is always 1.0
    df_filtered = df[df["method"] != "full"]

    if df_filtered.empty:
        plt.close(fig)
        return ""

    grouped = df_filtered.groupby("method").agg({
        "similarity_score": ["mean", "std"],
        "rouge_l_f1": ["mean"],
    })

    # Flatten column names
    grouped.columns = ["sim_mean", "sim_std", "rouge_mean"]
    grouped = _ordered_reindex(grouped)
    grouped = grouped.dropna()

    if grouped.empty:
        plt.close(fig)
        return ""

    methods = grouped.index.tolist()
    x = range(len(methods))
    width = 0.35

    ax.bar(
        [i - width/2 for i in x],
        grouped["sim_mean"],
        width,
        yerr=grouped["sim_std"],
        label="Semantic Similarity",
        color=_COLORS["purple"],
        alpha=0.85,
        capsize=5,
    )
    ax.bar(
        [i + width/2 for i in x],
        grouped["rouge_mean"],
        width,
        label="ROUGE-L F1",
        color=_COLORS["teal"],
        alpha=0.85,
    )

    ax.set_xlabel("Compression Method", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Answer Quality: Compressed vs Full Context",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels([m.capitalize() for m in methods])
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Full Context (baseline)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / "similarity.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved: {output_path}")
    return str(output_path)


def plot_comparison_table(df: pd.DataFrame, output_dir: str) -> str:
    """
    Generate a summary comparison table as an image.

    Args:
        df: DataFrame with benchmark metrics.
        output_dir: Directory to save the plot.

    Returns:
        Path to saved plot.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    summary = df.groupby("method").agg({
        "tokens_before": "mean",
        "tokens_after": "mean",
        "reduction_percent": "mean",
        "latency_total": "mean",
        "similarity_score": "mean",
        "rouge_l_f1": "mean",
    }).round(2)
    summary = _ordered_reindex(summary)

    if summary.empty:
        plt.close(fig)
        return ""

    col_labels = [
        "Method", "Avg Tokens\nBefore", "Avg Tokens\nAfter",
        "Reduction %", "Avg Latency\n(s)", "Semantic\nSimilarity", "ROUGE-L\nF1",
    ]

    table_data = []
    for method, row in summary.iterrows():
        table_data.append([
            method.capitalize(),
            f"{row['tokens_before']:.0f}",
            f"{row['tokens_after']:.0f}",
            f"{row['reduction_percent']:.1f}%",
            f"{row['latency_total']:.2f}",
            f"{row['similarity_score']:.3f}",
            f"{row['rouge_l_f1']:.3f}",
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header row
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(_COLORS["dark"])
        cell.set_text_props(color="white", fontweight="bold")

    ax.set_title(
        "Benchmark Summary: Context Compression Methods",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    output_path = Path(output_dir) / "summary_table.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved: {output_path}")
    return str(output_path)


def plot_summary_bars(df: pd.DataFrame, output_dir: str) -> str:
    """
    Generate a combined summary bar chart showing all key metrics
    side-by-side per method. This is the main overview chart.

    Args:
        df: DataFrame with benchmark metrics.
        output_dir: Directory to save the plot.

    Returns:
        Path to saved plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    summary = df.groupby("method").agg({
        "reduction_percent": "mean",
        "similarity_score": "mean",
        "latency_total": "mean",
    })
    summary = _ordered_reindex(summary)

    if summary.empty:
        plt.close(fig)
        return ""

    methods = [m.capitalize() for m in summary.index.tolist()]
    colors = ["#3498DB", "#E74C3C", "#2ECC71", "#E67E22", "#9B59B6"]
    colors = colors[:len(methods)]

    # Chart 1: Token Reduction
    axes[0].bar(methods, summary["reduction_percent"], color=colors, alpha=0.85)
    axes[0].set_title("Avg Token Reduction (%)", fontweight="bold")
    axes[0].set_ylabel("Reduction %")
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(summary["reduction_percent"]):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)

    # Chart 2: Semantic Similarity
    axes[1].bar(methods, summary["similarity_score"], color=colors, alpha=0.85)
    axes[1].set_title("Avg Semantic Similarity", fontweight="bold")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.1)
    axes[1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(summary["similarity_score"]):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    # Chart 3: Total Latency
    axes[2].bar(methods, summary["latency_total"], color=colors, alpha=0.85)
    axes[2].set_title("Avg Total Latency (s)", fontweight="bold")
    axes[2].set_ylabel("Seconds")
    axes[2].grid(axis="y", alpha=0.3)
    for i, v in enumerate(summary["latency_total"]):
        axes[2].text(i, v + 0.05, f"{v:.2f}s", ha="center", fontsize=9)

    # Rotate labels
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Benchmark Summary: All Methods Compared",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = Path(output_dir) / "summary_bars.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved: {output_path}")
    return str(output_path)


def generate_all_plots(
    csv_path: str,
    output_dir: str,
) -> None:
    """
    Generate all visualization plots from benchmark results.

    Args:
        csv_path: Path to metrics.csv.
        output_dir: Directory to save plots.
    """
    print("\nGenerating visualizations...")

    df = load_metrics(csv_path)
    print(f"  Loaded {len(df)} experiment results")

    plots_dir = str(Path(output_dir) / "plots")

    plot_token_reduction(df, plots_dir)
    plot_latency(df, plots_dir)
    plot_similarity(df, plots_dir)
    plot_comparison_table(df, plots_dir)
    plot_summary_bars(df, plots_dir)

    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    generate_all_plots(
        csv_path=str(project_root / "results" / "metrics.csv"),
        output_dir=str(project_root / "results"),
    )
