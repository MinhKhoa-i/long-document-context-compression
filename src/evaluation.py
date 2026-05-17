"""
Evaluation Metrics Module
=========================
Metrics for evaluating context compression effectiveness:
- Token reduction percentage
- Latency measurement
- Semantic answer similarity
- ROUGE-L (optional)
"""

import time
from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer, util


_eval_model: Optional[SentenceTransformer] = None


def _get_eval_model() -> SentenceTransformer:
    """Lazy-load the evaluation embedding model."""
    global _eval_model
    if _eval_model is None:
        _eval_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _eval_model


# ---------------------------------------------------------------------------
#  Token Counting
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """
    Approximate token count by splitting on whitespace.
    A simple but consistent proxy for actual LLM tokenization.

    Args:
        text: Input text.

    Returns:
        Approximate token count.
    """
    if not text:
        return 0
    return len(text.split())


def token_reduction(tokens_before: int, tokens_after: int) -> float:
    """
    Calculate token reduction percentage.

    Formula: Reduction% = ((before - after) / before) * 100

    Args:
        tokens_before: Token count before compression.
        tokens_after: Token count after compression.

    Returns:
        Reduction percentage (0-100).
    """
    if tokens_before == 0:
        return 0.0
    return ((tokens_before - tokens_after) / tokens_before) * 100.0


# ---------------------------------------------------------------------------
#  Latency Measurement
# ---------------------------------------------------------------------------

class LatencyTracker:
    """
    Context manager and utility for measuring pipeline stage latencies.

    Usage:
        tracker = LatencyTracker()
        with tracker.track("retrieval"):
            results = retriever.retrieve(query)
        print(tracker.get_all())
    """

    def __init__(self):
        self._timings: Dict[str, float] = {}
        self._current_stage: Optional[str] = None
        self._start_time: float = 0.0

    def track(self, stage: str) -> "LatencyTracker":
        """Return self as context manager for the given stage."""
        self._current_stage = stage
        return self

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self._start_time
        if self._current_stage:
            self._timings[self._current_stage] = elapsed
        return False

    def get(self, stage: str) -> float:
        """Get latency for a specific stage in seconds."""
        return self._timings.get(stage, 0.0)

    def get_all(self) -> Dict[str, float]:
        """Get all recorded latencies."""
        result = dict(self._timings)
        result["total"] = sum(self._timings.values())
        return result


# ---------------------------------------------------------------------------
#  Semantic Similarity
# ---------------------------------------------------------------------------

def semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two texts using embeddings.

    Args:
        text_a: First text (e.g., answer from full context).
        text_b: Second text (e.g., answer from compressed context).

    Returns:
        Cosine similarity score in [-1, 1].
    """
    if not text_a or not text_b:
        return 0.0

    model = _get_eval_model()
    emb_a = model.encode(text_a, convert_to_tensor=True, normalize_embeddings=True)
    emb_b = model.encode(text_b, convert_to_tensor=True, normalize_embeddings=True)

    score = util.cos_sim(emb_a, emb_b).item()
    return float(score)


# ---------------------------------------------------------------------------
#  ROUGE-L
# ---------------------------------------------------------------------------

def _lcs_length(x: List[str], y: List[str]) -> int:
    """Compute length of Longest Common Subsequence."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0

    # Space-optimized LCS
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(curr[j-1], prev[j])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]


def rouge_l(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Compute ROUGE-L score between reference and hypothesis.

    Args:
        reference: Reference text (ground truth or full-context answer).
        hypothesis: Hypothesis text (compressed-context answer).

    Returns:
        Dict with 'precision', 'recall', 'f1' keys.
    """
    if not reference or not hypothesis:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    lcs_len = _lcs_length(ref_tokens, hyp_tokens)

    precision = lcs_len / len(hyp_tokens) if hyp_tokens else 0.0
    recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
#  Combined Evaluation
# ---------------------------------------------------------------------------

def evaluate_compression(
    full_context: str,
    compressed_context: str,
    full_answer: str,
    compressed_answer: str,
    latencies: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Run all evaluation metrics for a single query comparison.

    Args:
        full_context: Full (uncompressed) context text.
        compressed_context: Compressed context text.
        full_answer: Answer generated from full context.
        compressed_answer: Answer generated from compressed context.
        latencies: Optional dict of measured latencies.

    Returns:
        Dictionary of all metric results.
    """
    tokens_before = count_tokens(full_context)
    tokens_after = count_tokens(compressed_context)
    reduction = token_reduction(tokens_before, tokens_after)

    sim_score = semantic_similarity(full_answer, compressed_answer)
    rouge = rouge_l(full_answer, compressed_answer)

    result = {
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "reduction_percent": round(reduction, 2),
        "similarity_score": round(sim_score, 4),
        "rouge_l_f1": round(rouge["f1"], 4),
        "rouge_l_precision": round(rouge["precision"], 4),
        "rouge_l_recall": round(rouge["recall"], 4),
    }

    if latencies:
        result["latency"] = latencies

    return result
