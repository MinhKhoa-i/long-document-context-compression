"""
Semantic Retrieval Module
=========================
Embedding-based retrieval for document chunks.
Uses sentence-transformers for dense vector search.
"""

from typing import List

from sentence_transformers import SentenceTransformer, util

from models import RetrievedChunk


class SemanticRetriever:
    """
    Dense semantic retriever using sentence-transformer embeddings.
    Supports loading chunks from structured records (PDF pipeline)
    or from raw chunk lists.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)
        self.chunks: List[RetrievedChunk] = []
        self.embeddings = None

    def load_chunks(self, chunk_records: List[dict]) -> List[RetrievedChunk]:
        """
        Load chunks from structured records (output of chunking.chunk_document).

        Args:
            chunk_records: List of dicts with keys: doc_id, chunk_id, text, source.

        Returns:
            List of RetrievedChunk objects.
        """
        self.chunks = []
        for rec in chunk_records:
            if not rec.get("text", "").strip():
                continue
            self.chunks.append(
                RetrievedChunk(
                    doc_id=rec["doc_id"],
                    chunk_id=rec["chunk_id"],
                    text=rec["text"],
                    source=rec.get("source", ""),
                    doc_type=rec.get("doc_type", "paper"),
                )
            )
        return self.chunks

    def load_retrieved_chunks(self, chunks: List[RetrievedChunk]) -> None:
        """
        Directly load a list of RetrievedChunk objects.

        Args:
            chunks: Pre-built chunk objects.
        """
        self.chunks = [c for c in chunks if c.text.strip()]

    def build_index(self) -> None:
        """Build the embedding index from loaded chunks."""
        if not self.chunks:
            raise ValueError("No chunks loaded. Call load_chunks() first.")

        texts = [c.text for c in self.chunks]
        self.embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        print(f"  Index built: {len(self.chunks)} chunks embedded")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        """
        Retrieve top-k most relevant chunks for a query.

        Args:
            query: The search query.
            top_k: Number of results to return.

        Returns:
            List of RetrievedChunk objects sorted by relevance score.
        """
        if self.embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")

        query_emb = self.model.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        scores = util.cos_sim(query_emb, self.embeddings)[0]
        top_results = scores.topk(k=min(top_k, len(self.chunks)))

        results: List[RetrievedChunk] = []
        for score, idx in zip(
            top_results.values.tolist(),
            top_results.indices.tolist(),
        ):
            chunk = RetrievedChunk(
                doc_id=self.chunks[idx].doc_id,
                chunk_id=self.chunks[idx].chunk_id,
                text=self.chunks[idx].text,
                source=self.chunks[idx].source,
                doc_type=self.chunks[idx].doc_type,
                retrieval_score=float(score),
            )
            results.append(chunk)

        results.sort(key=lambda x: x.retrieval_score, reverse=True)
        return results