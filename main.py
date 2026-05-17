"""
Context Compression for Long Documents in LLMs
================================================
Main entry point for the research pipeline.

Usage:
    # Run interactive QA
    python main.py

    # Run full benchmark
    python main.py --benchmark

    # Generate plots from existing results
    python main.py --plot

    # Print summary table
    python main.py --summary
"""

import argparse
import sys
from pathlib import Path

# Ensure src is importable
_project_root = Path(__file__).resolve().parent
_src_dir = _project_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))


def run_interactive(papers_dir: str) -> None:
    """Run interactive QA pipeline."""
    from pipeline import QAPipeline
    from compressor import get_available_methods

    pipeline = QAPipeline()
    pipeline.load_papers(papers_dir)

    methods = get_available_methods()
    print(f"\nAvailable methods: {methods}")
    print("Type 'exit' to quit, 'method <name>' to switch method.\n")

    current_method = "hybrid"

    while True:
        query = input(f"[{current_method}] Question: ").strip()

        if query.lower() == "exit":
            break
        if not query:
            continue
        if query.startswith("method "):
            new_method = query.split(" ", 1)[1].strip()
            if new_method in methods:
                current_method = new_method
                print(f"  Switched to: {current_method}")
            else:
                print(f"  Unknown method. Available: {methods}")
            continue

        result = pipeline.ask(query, method=current_method)

        print(f"\n{'=' * 70}")
        print(f"Method: {result['method']}")
        print(f"Tokens: {result['tokens_before']} -> {result['tokens_after']} "
              f"({result['reduction_percent']}% reduction)")
        print(f"Latency: {result['latency']}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"{'=' * 70}\n")


def run_benchmark(
    papers_dir: str,
    questions_path: str,
    results_dir: str,
) -> None:
    """Run full benchmark experiment."""
    from benchmark import run_benchmark as _run_benchmark

    results = _run_benchmark(
        papers_dir=papers_dir,
        questions_path=questions_path,
        results_dir=results_dir,
    )

    if results:
        print(f"\n{len(results)} experiments completed.")

        # Auto-generate plots
        try:
            from visualization import generate_all_plots
            generate_all_plots(
                csv_path=str(Path(results_dir) / "metrics.csv"),
                output_dir=results_dir,
            )
        except Exception as e:
            print(f"[WARNING] Plot generation failed: {e}")


def run_plots(results_dir: str) -> None:
    """Generate plots from existing results."""
    from visualization import generate_all_plots
    generate_all_plots(
        csv_path=str(Path(results_dir) / "metrics.csv"),
        output_dir=results_dir,
    )


def run_summary(results_dir: str) -> None:
    """Print summary table from existing results."""
    import pandas as pd

    csv_path = Path(results_dir) / "metrics.csv"
    if not csv_path.exists():
        print(f"[ERROR] No results found at {csv_path}")
        print("Run benchmark first: python main.py --benchmark")
        return

    df = pd.read_csv(csv_path)
    summary = df.groupby("method").agg({
        "tokens_before": "mean",
        "tokens_after": "mean",
        "reduction_percent": "mean",
        "latency_total": "mean",
        "similarity_score": "mean",
        "rouge_l_f1": "mean",
    }).round(3)

    method_order = ["full", "extractive", "mmr", "summary", "hybrid"]
    summary = summary.reindex([m for m in method_order if m in summary.index])

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(summary.to_string())
    print("=" * 70)
    print(f"\nTotal experiments: {len(df)}")
    print(f"Papers: {df['paper'].nunique()}")
    print(f"Questions: {len(df) // df['method'].nunique()}")


def main():
    parser = argparse.ArgumentParser(
        description="Context Compression for Long Documents in LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Interactive QA mode
  python main.py --benchmark        # Run full benchmark
  python main.py --plot             # Generate plots from results
  python main.py --summary          # Print summary table
        """,
    )

    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run full benchmark across all papers, questions, and methods",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate visualization plots from existing results",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print summary table from existing benchmark results",
    )
    parser.add_argument(
        "--papers-dir", type=str, default=None,
        help="Path to papers directory (default: data/papers)",
    )
    parser.add_argument(
        "--questions", type=str, default=None,
        help="Path to questions.json (default: data/questions.json)",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Path to results directory (default: results/)",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    papers_dir = args.papers_dir or str(project_root / "data" / "papers")
    questions_path = args.questions or str(project_root / "data" / "questions.json")
    results_dir = args.results_dir or str(project_root / "results")

    if args.summary:
        run_summary(results_dir)
    elif args.plot:
        run_plots(results_dir)
    elif args.benchmark:
        run_benchmark(papers_dir, questions_path, results_dir)
    else:
        run_interactive(papers_dir)


if __name__ == "__main__":
    main()

