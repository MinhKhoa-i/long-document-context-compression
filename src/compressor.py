"""
Context Compression Module
===========================
Multiple compression strategies for reducing retrieved context
before sending to LLM. Core contribution of this research project.

Methods:
    - full: No compression, use all retrieved chunks (baseline).
    - extractive: Query-aware sentence extraction.
    - mmr: Maximal Marginal Relevance for diversity-aware selection.
    - summary: LLM-based summarization of retrieved context.
    - hybrid: Extractive + MMR combined approach.
"""

import re
import time
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, util

from models import RetrievedChunk

# Shared embedding model for compression scoring
_sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
#  Text Utilities
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    """
    Split text into sentence-level units.
    Handles bullet points, headings, and regular sentences.

    Args:
        text: Input text to split.

    Returns:
        List of sentence strings.
    """
    units: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Skip very short markdown headings
        if line.startswith("#") and len(line) < 40:
            continue
        # Bullet / list items -> keep as single unit
        if re.match(r"^[-*•]\s+", line):
            units.append(line)
            continue
        # Split by sentence boundaries
        parts = re.split(r"(?<=[.!?])\s+", line)
        for p in parts:
            p = p.strip()
            if p:
                units.append(p)

    return units


def _clean(text: str) -> str:
    """Normalize whitespace in text."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
#  Compression Methods
# ---------------------------------------------------------------------------

def _compress_full(
    query: str,
    retrieved_chunks: List[RetrievedChunk],
    **kwargs,
) -> Tuple[str, Dict]:
    """
    Full context baseline — no compression applied.
    Concatenates all retrieved chunks as-is.
    """
    parts = []
    for i, chunk in enumerate(retrieved_chunks):
        parts.append(f"[{i+1}] {chunk.text}")

    compressed = "\n\n".join(parts)
    meta = {"method": "full", "units_selected": len(retrieved_chunks)}
    return compressed, meta


def _compress_extractive(
    query: str,
    retrieved_chunks: List[RetrievedChunk],
    max_sentences: int = 15,
    **kwargs,
) -> Tuple[str, Dict]:
    """
    Extractive compression — select top sentences by query similarity.
    Preserves original wording, removes less relevant content.
    """
    query_emb = _sentence_model.encode(
        query, convert_to_tensor=True, normalize_embeddings=True,
    )

    candidates = []
    for chunk in retrieved_chunks:
        units = [_clean(u) for u in split_sentences(chunk.text)]
        units = [u for u in units if len(u) >= 40]
        if not units:
            continue

        unit_embs = _sentence_model.encode(
            units, convert_to_tensor=True, normalize_embeddings=True,
        )
        sims = util.cos_sim(query_emb, unit_embs)[0]

        for unit, sim in zip(units, sims.tolist()):
            score = float(sim) + 0.2 * chunk.retrieval_score
            candidates.append((score, unit, chunk))

    candidates.sort(key=lambda x: x[0], reverse=True)

    selected = []
    seen = set()
    for score, unit, chunk in candidates:
        key = unit.lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append(f"{chunk.chunk_id}\n{unit}")
        if len(selected) >= max_sentences:
            break

    compressed = "\n\n".join(
        f"[{i}] {item}" for i, item in enumerate(selected, start=1)
    )
    meta = {"method": "extractive", "units_selected": len(selected)}
    return compressed, meta


def _compress_mmr(
    query: str,
    retrieved_chunks: List[RetrievedChunk],
    max_sentences: int = 15,
    lambda_param: float = 0.7,
    **kwargs,
) -> Tuple[str, Dict]:
    """
    MMR (Maximal Marginal Relevance) compression.
    Balances relevance to query with diversity among selected sentences.
    """
    query_emb = _sentence_model.encode(
        query, convert_to_tensor=True, normalize_embeddings=True,
    )

    all_units = []
    all_chunk_refs = []
    for chunk in retrieved_chunks:
        units = [_clean(u) for u in split_sentences(chunk.text)]
        units = [u for u in units if len(u) >= 15]
        for u in units:
            all_units.append(u)
            all_chunk_refs.append(chunk)

    if not all_units:
        return "", {"method": "mmr", "units_selected": 0}

    unit_embs = _sentence_model.encode(
        all_units, convert_to_tensor=True, normalize_embeddings=True,
    )
    query_sims = util.cos_sim(query_emb, unit_embs)[0].cpu().numpy()
    unit_unit_sims = util.cos_sim(unit_embs, unit_embs).cpu().numpy()

    selected_indices = []
    remaining = set(range(len(all_units)))

    for _ in range(min(max_sentences, len(all_units))):
        best_idx = -1
        best_score = -float("inf")

        for idx in remaining:
            relevance = query_sims[idx]
            if selected_indices:
                max_redundancy = max(
                    unit_unit_sims[idx][s] for s in selected_indices
                )
            else:
                max_redundancy = 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx == -1:
            break
        selected_indices.append(best_idx)
        remaining.discard(best_idx)

    parts = []
    for rank, idx in enumerate(selected_indices, start=1):
        chunk = all_chunk_refs[idx]
        parts.append(f"[{rank}] {chunk.chunk_id}\n{all_units[idx]}")

    compressed = "\n\n".join(parts)
    meta = {"method": "mmr", "units_selected": len(selected_indices)}
    return compressed, meta


def _compress_summary(
    query: str,
    retrieved_chunks: List[RetrievedChunk],
    **kwargs,
) -> Tuple[str, Dict]:
    """
    Summary compression — uses LLM to generate a focused summary.
    Falls back to extractive if LLM is unavailable.
    """
    from generator import generate_answer

    full_context = "\n\n".join(c.text for c in retrieved_chunks)

    summary_prompt = f"""Summarize the following context focusing on information 
relevant to the question. Keep only the most important facts and findings.
Be concise but preserve key details, numbers, and technical terms.

Question: {query}

Context:
{full_context}

Focused Summary:"""

    try:
        summary = generate_answer(summary_prompt)
        meta = {"method": "summary", "units_selected": 1}
        return summary, meta
    except Exception as e:
        print(f"  [WARNING] Summary compression failed ({e}), falling back to extractive")
        return _compress_extractive(query, retrieved_chunks, **kwargs)


def _compress_hybrid(
    query: str,
    retrieved_chunks: List[RetrievedChunk],
    max_sentences: int = 15,
    **kwargs,
) -> Tuple[str, Dict]:
    """
    Hybrid compression — Extractive scoring + MMR diversity filtering.
    First scores sentences by query relevance, then applies MMR
    to reduce redundancy among top candidates.
    """
    query_emb = _sentence_model.encode(
        query, convert_to_tensor=True, normalize_embeddings=True,
    )

    # Step 1: Collect and score all units (extractive)
    all_units = []
    all_scores = []
    all_chunk_refs = []

    for chunk in retrieved_chunks:
        units = [_clean(u) for u in split_sentences(chunk.text)]
        units = [u for u in units if len(u) >= 40]
        if not units:
            continue

        unit_embs = _sentence_model.encode(
            units, convert_to_tensor=True, normalize_embeddings=True,
        )
        sims = util.cos_sim(query_emb, unit_embs)[0]

        for unit, sim in zip(units, sims.tolist()):
            score = float(sim) + 0.2 * chunk.retrieval_score
            all_units.append(unit)
            all_scores.append(score)
            all_chunk_refs.append(chunk)

    if not all_units:
        return "", {"method": "hybrid", "units_selected": 0}

    # Step 2: Pre-filter to top 2x candidates by relevance
    n_candidates = min(max_sentences * 2, len(all_units))
    sorted_indices = np.argsort(all_scores)[::-1][:n_candidates]

    candidate_units = [all_units[i] for i in sorted_indices]
    candidate_refs = [all_chunk_refs[i] for i in sorted_indices]

    # Step 3: Apply MMR on filtered candidates
    cand_embs = _sentence_model.encode(
        candidate_units, convert_to_tensor=True, normalize_embeddings=True,
    )
    query_sims = util.cos_sim(query_emb, cand_embs)[0].cpu().numpy()
    cand_sims = util.cos_sim(cand_embs, cand_embs).cpu().numpy()

    selected_indices = []
    remaining = set(range(len(candidate_units)))
    lambda_param = 0.85

    for _ in range(min(max_sentences, len(candidate_units))):
        best_idx = -1
        best_score = -float("inf")

        for idx in remaining:
            relevance = query_sims[idx]
            if selected_indices:
                max_redundancy = max(
                    cand_sims[idx][s] for s in selected_indices
                )
            else:
                max_redundancy = 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx == -1:
            break
        selected_indices.append(best_idx)
        remaining.discard(best_idx)

    parts = []
    seen = set()
    for rank, idx in enumerate(selected_indices, start=1):
        unit = candidate_units[idx]
        key = unit.lower()
        if key in seen:
            continue
        seen.add(key)
        chunk = candidate_refs[idx]
        parts.append(f"[{rank}] {chunk.chunk_id}\n{unit}")

    compressed = "\n\n".join(parts)
    meta = {"method": "hybrid", "units_selected": len(parts)}
    return compressed, meta


# ---------------------------------------------------------------------------
#  Unified Interface
# ---------------------------------------------------------------------------

_METHODS = {
    "full": _compress_full,
    "extractive": _compress_extractive,
    "mmr": _compress_mmr,
    "summary": _compress_summary,
    "hybrid": _compress_hybrid,
}


def compress_context(
    query: str,
    retrieved_chunks: List[RetrievedChunk],
    method: str = "hybrid",
    compression_ratio: float = 0.35,
    max_sentences: int = 20,
    **kwargs,
) -> Tuple[str, Dict]:
    """
    Unified context compression interface.

    Args:
        query: User query for relevance scoring.
        retrieved_chunks: List of retrieved chunks to compress.
        method: Compression method — one of:
            'full', 'extractive', 'mmr', 'summary', 'hybrid'.
        compression_ratio: Target compression ratio (0-1).
            Used to adjust max_sentences relative to total content.
        max_sentences: Maximum number of sentences to keep.
        **kwargs: Additional method-specific parameters.

    Returns:
        Tuple of (compressed_text, metadata_dict).
    """
    if method not in _METHODS:
        raise ValueError(
            f"Unknown compression method: '{method}'. "
            f"Available: {list(_METHODS.keys())}"
        )

    # Adjust max_sentences based on compression_ratio for non-full methods
    if method != "full" and compression_ratio < 1.0:
        total_text = " ".join(c.text for c in retrieved_chunks)
        total_sentences = len(split_sentences(total_text))
        adjusted = max(8, int(total_sentences * compression_ratio))
        if max_sentences is None:
            max_sentences = adjusted
        else:
            max_sentences = max(max_sentences, adjusted)

    t0 = time.time()
    compressed, meta = _METHODS[method](
        query=query,
        retrieved_chunks=retrieved_chunks,
        max_sentences=max_sentences,
        **kwargs,
    )
    meta["compression_time"] = time.time() - t0

    return compressed, meta


def get_available_methods() -> List[str]:
    """Return list of available compression method names."""
    return list(_METHODS.keys())