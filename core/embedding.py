"""Embedding utilities using OpenRouter's OpenAI-compatible embeddings endpoint."""

import os
import time
import requests
import numpy as np
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
BATCH_SIZE = 100
SLEEP_PER_BATCH = 0.1


def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a vector. Returns the original if norm is near zero."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > eps else vec


def embed_texts(texts: list[str], model: str = EMBEDDING_MODEL) -> list[np.ndarray]:
    """Embed a list of texts in batches via OpenRouter. Returns L2-normalized vectors."""
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = requests.post(
            OPENROUTER_EMBEDDINGS_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"input": batch, "model": model},
        )
        if response.status_code != 200:
            raise RuntimeError(f"Embedding API error {response.status_code}: {response.text}")

        for item in response.json()["data"]:
            vec = np.array(item["embedding"], dtype=np.float32)
            embeddings.append(_l2_normalize(vec))
        time.sleep(SLEEP_PER_BATCH)
    return embeddings


def embed_query(query: str, model: str = EMBEDDING_MODEL) -> np.ndarray:
    """Embed a single query string. Returns an L2-normalized vector."""
    return embed_texts([query], model=model)[0]
