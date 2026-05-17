"""
LLM Answer Generator Module
============================
Generates answers using Ollama local LLM.
Designed for scientific paper QA.
"""

import requests
from typing import Optional

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b-instruct"


def build_prompt(query: str, compressed_context: str) -> str:
    """
    Build a prompt for scientific paper QA.

    Args:
        query: User question about the paper(s).
        compressed_context: Compressed/retrieved context from papers.

    Returns:
        Formatted prompt string.
    """
    return f"""You are a research assistant that answers questions about scientific papers.

Rules:
- Answer ONLY based on the provided context.
- Do NOT use external knowledge beyond the context.
- If the context contains the answer, respond directly and concisely.
- If the context does NOT contain enough information, say: "The provided context does not contain sufficient information to answer this question."
- Cite specific findings, methods, or results from the context when possible.

[QUESTION]
{query}

[CONTEXT]
{compressed_context}

[ANSWER]
"""


def generate_answer(
    prompt: str,
    model: str = OLLAMA_MODEL,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """
    Generate an answer using Ollama LLM.

    Args:
        prompt: The full prompt to send.
        model: Ollama model name.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = deterministic).

    Returns:
        Generated answer text.

    Raises:
        RuntimeError: If Ollama is unreachable or returns an error.
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=120,
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama at http://localhost:11434. "
            "Please run: ollama serve"
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama HTTP error {response.status_code}: {response.text}"
        )

    data = response.json()

    if "message" not in data or "content" not in data["message"]:
        raise RuntimeError(f"Invalid Ollama response: {data}")

    return data["message"]["content"].strip()