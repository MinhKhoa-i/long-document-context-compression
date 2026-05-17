"""
Document Chunking Module
========================
Split long document text into overlapping chunks suitable for
embedding-based retrieval. Designed for scientific papers.
"""

from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[str]:
    """
    Split text into overlapping chunks based on word count.

    Args:
        text: Raw document text to chunk.
        chunk_size: Target number of words per chunk.
        overlap: Number of overlapping words between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()]

    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk.strip())

        if end >= len(words):
            break
        start += step

    return chunks


def chunk_document(
    text: str,
    filename: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[dict]:
    """
    Chunk a document and return structured chunk records.

    Args:
        text: Raw document text.
        filename: Source filename for metadata.
        chunk_size: Target number of words per chunk.
        overlap: Number of overlapping words between consecutive chunks.

    Returns:
        List of dicts with keys: doc_id, chunk_id, text, source.
    """
    raw_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    records = []
    for i, chunk in enumerate(raw_chunks):
        records.append({
            "doc_id": filename,
            "chunk_id": f"{filename}_chunk_{i:03d}",
            "text": chunk,
            "source": filename,
        })

    return records


if __name__ == "__main__":
    # Quick test
    sample = "word " * 1200  # 1200 words
    chunks = chunk_text(sample, chunk_size=500, overlap=50)
    print(f"Input: {len(sample.split())} words")
    print(f"Chunks: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"  Chunk {i}: {len(c.split())} words")
