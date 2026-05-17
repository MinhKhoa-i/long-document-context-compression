"""
Benchmark Runner
================
Automated experiment runner that iterates across:
- All papers (PDFs)
- All questions per paper
- All compression methods

Produces CSV metrics, JSON outputs, and summary CSV for analysis.
"""

import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path for imports
_src_dir = Path(__file__).resolve().parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from pdf_loader import load_all_pdfs
from chunking import chunk_document
from retrieval import SemanticRetriever
from compressor import compress_context, get_available_methods
from generator import build_prompt, generate_answer
from evaluation import (
    count_tokens,
    token_reduction,
    semantic_similarity,
    rouge_l,
    LatencyTracker,
)


def load_questions(questions_path: str) -> Dict[str, List[str]]:
    """
    Load question dataset from JSON file.

    Args:
        questions_path: Path to questions.json.

    Returns:
        Dict mapping paper filename -> list of questions.
    """
    path = Path(questions_path)
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _generate_full_baseline(
    query: str,
    retrieved,
    tracker: LatencyTracker,
) -> Dict:
    """
    Generate the full-context baseline answer for a query.
    Called once per question, reused across all compression methods.

    Returns:
        Dict with full_context, full_answer, tokens_before, and latency.
    """
    full_context = "\n\n".join(c.text for c in retrieved)
    tokens_before = count_tokens(full_context)

    full_prompt = build_prompt(query, full_context)
    with tracker.track("generation_full"):
        try:
            full_answer = generate_answer(full_prompt)
        except Exception as e:
            full_answer = f"[Generation Error: {e}]"

    return {
        "full_context": full_context,
        "full_answer": full_answer,
        "tokens_before": tokens_before,
    }


def run_single_experiment(
    query: str,
    paper_name: str,
    retriever: SemanticRetriever,
    method: str,
    top_k: int = 5,
    baseline: Optional[Dict] = None,
) -> Dict:
    """
    Run a single experiment: retrieve -> compress -> generate -> evaluate.

    Args:
        query: The question to answer.
        paper_name: Name of the source paper.
        retriever: Pre-built SemanticRetriever with indexed chunks.
        method: Compression method to use.
        top_k: Number of chunks to retrieve.
        baseline: Pre-computed full-context baseline (avoids redundant LLM calls).

    Returns:
        Dictionary with all results and metrics.
    """
    tracker = LatencyTracker()

    # Step 1: Retrieval
    with tracker.track("retrieval"):
        retrieved = retriever.retrieve(query, top_k=top_k)

    if not retrieved:
        return {
            "paper": paper_name,
            "question": query,
            "method": method,
            "error": "No chunks retrieved",
        }

    # Step 2: Get full context baseline
    if baseline is None:
        baseline = _generate_full_baseline(query, retrieved, tracker)

    full_context = baseline["full_context"]
    full_answer = baseline["full_answer"]
    tokens_before = baseline["tokens_before"]

    # Step 3: Compression
    with tracker.track("compression"):
        compressed_context, comp_meta = compress_context(
            query=query,
            retrieved_chunks=retrieved,
            method=method,
        )
    tokens_after = count_tokens(compressed_context)

    # Step 4: Generate answer from compressed context
    if method == "full":
        # For full method, answer is the baseline answer
        answer = full_answer
    else:
        prompt = build_prompt(query, compressed_context)
        with tracker.track("generation"):
            try:
                answer = generate_answer(prompt)
            except Exception as e:
                answer = f"[Generation Error: {e}]"

    # Step 5: Compute metrics
    reduction = token_reduction(tokens_before, tokens_after)
    sim_score = semantic_similarity(full_answer, answer) if method != "full" else 1.0
    rouge = rouge_l(full_answer, answer) if method != "full" else {"f1": 1.0}

    latencies = tracker.get_all()

    return {
        "paper": paper_name,
        "question": query,
        "method": method,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "reduction_percent": round(reduction, 2),
        "latency_retrieval": round(latencies.get("retrieval", 0), 4),
        "latency_compression": round(latencies.get("compression", 0), 4),
        "latency_generation": round(latencies.get("generation", 0), 4),
        "latency_total": round(latencies.get("total", 0), 4),
        "similarity_score": round(sim_score, 4),
        "rouge_l_f1": round(rouge.get("f1", 0), 4),
        "answer": answer,
        "full_answer": full_answer,
        "compressed_context_preview": compressed_context[:500],
    }


def run_benchmark(
    papers_dir: str,
    questions_path: str,
    results_dir: str,
    methods: Optional[List[str]] = None,
    top_k: int = 5,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict]:
    """
    Run the full benchmark across all papers, questions, and methods.

    For each question, the full-context baseline is generated once and
    reused across all compression methods to avoid redundant LLM calls.

    Args:
        papers_dir: Path to directory containing PDF papers.
        questions_path: Path to questions.json.
        results_dir: Path to output directory for results.
        methods: List of compression methods to test (default: all).
        top_k: Number of chunks to retrieve.
        chunk_size: Words per chunk.
        chunk_overlap: Overlap words between chunks.

    Returns:
        List of all experiment result dictionaries.
    """
    if methods is None:
        methods = get_available_methods()

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Load questions
    print("=" * 60)
    print("BENCHMARK: Context Compression for Scientific Papers")
    print("=" * 60)

    questions = load_questions(questions_path)
    print(f"\nQuestions loaded: {sum(len(v) for v in questions.values())} "
          f"across {len(questions)} papers")
    print(f"Methods: {methods}")

    # Load and process PDFs
    print(f"\nLoading papers from: {papers_dir}")
    papers = load_all_pdfs(papers_dir)

    if not papers:
        print("[ERROR] No papers found. Place PDFs in the papers directory.")
        return []

    # Build chunks and index
    print(f"\nChunking documents (size={chunk_size}, overlap={chunk_overlap})...")
    all_chunks = []
    for filename, text in papers.items():
        doc_chunks = chunk_document(
            text, filename,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )
        all_chunks.extend(doc_chunks)
        print(f"  {filename}: {len(doc_chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    retriever = SemanticRetriever()
    retriever.load_chunks(all_chunks)
    retriever.build_index()

    # Run experiments
    all_results: List[Dict] = []
    total_experiments = 0

    for paper_file, paper_questions in questions.items():
        if paper_file not in papers:
            print(f"\n[SKIP] {paper_file} not found in papers directory")
            continue

        print(f"\n{'─' * 50}")
        print(f"Paper: {paper_file}")
        print(f"Questions: {len(paper_questions)}")

        for q_idx, question in enumerate(paper_questions):
            # Generate full-context baseline ONCE per question
            tracker = LatencyTracker()
            with tracker.track("retrieval"):
                retrieved = retriever.retrieve(question, top_k=top_k)

            if not retrieved:
                print(f"\n  [SKIP] No chunks retrieved for: {question[:60]}...")
                continue

            baseline = _generate_full_baseline(question, retrieved, tracker)
            print(f"\n  Q{q_idx+1}: {question[:70]}...")
            print(f"       Baseline tokens: {baseline['tokens_before']}")

            for method in methods:
                total_experiments += 1

                result = run_single_experiment(
                    query=question,
                    paper_name=paper_file,
                    retriever=retriever,
                    method=method,
                    top_k=top_k,
                    baseline=baseline,
                )
                all_results.append(result)

                # Progress summary
                if "error" not in result:
                    print(f"       [{method:>11s}] "
                          f"{result['tokens_before']} → {result['tokens_after']} tokens "
                          f"({result['reduction_percent']:5.1f}% reduction) | "
                          f"sim={result['similarity_score']:.3f}")

    # Save results
    _save_results(all_results, results_path)
    _save_summary(all_results, results_path)

    print(f"\n{'=' * 60}")
    print(f"BENCHMARK COMPLETE: {total_experiments} experiments")
    print(f"Results saved to: {results_path}")
    print(f"{'=' * 60}")

    return all_results


def _save_results(results: List[Dict], results_path: Path) -> None:
    """Save benchmark results to CSV and JSON files."""

    # Save CSV metrics
    csv_path = results_path / "metrics.csv"
    csv_columns = [
        "paper", "question", "method",
        "tokens_before", "tokens_after", "reduction_percent",
        "latency_retrieval", "latency_compression",
        "latency_generation", "latency_total",
        "similarity_score", "rouge_l_f1",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            if "error" not in r:
                writer.writerow(r)

    print(f"\n  Metrics saved: {csv_path}")

    # Save full outputs JSON
    json_path = results_path / "outputs.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"  Outputs saved: {json_path}")


def _save_summary(results: List[Dict], results_path: Path) -> None:
    """
    Generate and save a summary CSV with average metrics per method.
    This is the main comparison table for the research paper.
    """
    import pandas as pd

    valid = [r for r in results if "error" not in r]
    if not valid:
        return

    df = pd.DataFrame(valid)

    summary = df.groupby("method").agg({
        "tokens_before": "mean",
        "tokens_after": "mean",
        "reduction_percent": "mean",
        "latency_total": "mean",
        "similarity_score": "mean",
        "rouge_l_f1": "mean",
    }).round(3)

    # Reorder methods logically
    method_order = ["full", "extractive", "mmr", "summary", "hybrid"]
    summary = summary.reindex([m for m in method_order if m in summary.index])

    summary_path = results_path / "summary.csv"
    summary.to_csv(summary_path)

    print(f"  Summary saved: {summary_path}")
    print(f"\n  {'─' * 50}")
    print(f"  SUMMARY TABLE:")
    print(f"  {'─' * 50}")
    print(summary.to_string())
    print(f"  {'─' * 50}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    run_benchmark(
        papers_dir=str(project_root / "data" / "papers"),
        questions_path=str(project_root / "data" / "questions.json"),
        results_dir=str(project_root / "results"),
        methods=None,  # all methods
        top_k=5,
        chunk_size=500,
        chunk_overlap=50,
    )
