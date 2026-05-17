"""
Pipeline Module
===============
End-to-end pipeline for scientific paper QA with context compression.

Flow: PDF -> Text Extraction -> Chunking -> Retrieval -> Compression -> QA Generation
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Ensure src is importable
_src_dir = Path(__file__).resolve().parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from pdf_loader import load_pdf, load_all_pdfs
from chunking import chunk_document
from retrieval import SemanticRetriever
from compressor import compress_context, get_available_methods
from generator import build_prompt, generate_answer
from evaluation import count_tokens, LatencyTracker


class QAPipeline:
    """
    End-to-end QA pipeline for scientific papers with context compression.

    Usage:
        pipeline = QAPipeline()
        pipeline.load_papers("data/papers")
        result = pipeline.ask("What is self-attention?", method="hybrid")
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 5,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self.retriever = SemanticRetriever()
        self.papers: Dict[str, str] = {}
        self._ready = False

    def load_papers(self, papers_dir: str) -> None:
        """
        Load all PDFs from a directory, chunk them, and build index.

        Args:
            papers_dir: Path to directory containing PDF files.
        """
        print(f"Loading papers from: {papers_dir}")
        self.papers = load_all_pdfs(papers_dir)

        if not self.papers:
            raise ValueError(f"No PDFs found in {papers_dir}")

        # Chunk all papers
        all_chunks = []
        for filename, text in self.papers.items():
            doc_chunks = chunk_document(
                text, filename,
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap,
            )
            all_chunks.extend(doc_chunks)

        print(f"Total chunks: {len(all_chunks)}")

        # Build retrieval index
        self.retriever.load_chunks(all_chunks)
        self.retriever.build_index()
        self._ready = True

    def load_single_pdf(self, pdf_path: str) -> None:
        """
        Load a single PDF, chunk it, and build index.

        Args:
            pdf_path: Path to a PDF file.
        """
        path = Path(pdf_path)
        print(f"Loading: {path.name}")
        text = load_pdf(str(path))
        self.papers[path.name] = text

        doc_chunks = chunk_document(
            text, path.name,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )

        self.retriever.load_chunks(doc_chunks)
        self.retriever.build_index()
        self._ready = True

    def ask(
        self,
        query: str,
        method: str = "hybrid",
        top_k: Optional[int] = None,
    ) -> Dict:
        """
        Ask a question and get an answer with compression metrics.

        Args:
            query: The question about the paper(s).
            method: Compression method ('full', 'extractive', 'mmr', 'summary', 'hybrid').
            top_k: Override number of chunks to retrieve.

        Returns:
            Dictionary with answer, context, and metrics.
        """
        if not self._ready:
            raise RuntimeError("Pipeline not ready. Call load_papers() first.")

        k = top_k or self.top_k
        tracker = LatencyTracker()

        # Retrieve
        with tracker.track("retrieval"):
            retrieved = self.retriever.retrieve(query, top_k=k)

        # Full context (for comparison)
        full_context = "\n\n".join(c.text for c in retrieved)
        tokens_before = count_tokens(full_context)

        # Compress
        with tracker.track("compression"):
            compressed, comp_meta = compress_context(
                query=query,
                retrieved_chunks=retrieved,
                method=method,
            )
        tokens_after = count_tokens(compressed)

        # Generate
        prompt = build_prompt(query, compressed)
        with tracker.track("generation"):
            answer = generate_answer(prompt)

        latencies = tracker.get_all()

        return {
            "question": query,
            "method": method,
            "answer": answer,
            "compressed_context": compressed,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "reduction_percent": round(
                ((tokens_before - tokens_after) / tokens_before * 100)
                if tokens_before > 0 else 0, 2
            ),
            "latency": latencies,
            "compression_meta": comp_meta,
        }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    papers_dir = project_root / "data" / "papers"

    pipeline = QAPipeline()
    pipeline.load_papers(str(papers_dir))

    print("\nPipeline ready. Enter questions (type 'exit' to quit).\n")

    while True:
        query = input("Question: ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        result = pipeline.ask(query, method="hybrid")

        print(f"\n{'=' * 70}")
        print(f"Method: {result['method']}")
        print(f"Tokens: {result['tokens_before']} -> {result['tokens_after']} "
              f"({result['reduction_percent']}% reduction)")
        print(f"Latency: {result['latency']}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"{'=' * 70}\n")