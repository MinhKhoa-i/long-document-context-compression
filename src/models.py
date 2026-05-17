"""
Data Models
===========
Shared data structures used across the pipeline.
"""

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    """Represents a single chunk retrieved from the document index."""
    doc_id: str
    chunk_id: str
    text: str
    retrieval_score: float = 0.0
    rerank_score: float = 0.0
    source: str = ""
    doc_type: str = "paper"