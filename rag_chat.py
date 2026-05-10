"""
RAG chatbot for Airbnb FAQ data.

Embeds FAQ questions into a vector store, retrieves the top-k most similar
entries for a user query, and generates a grounded answer via an LLM.
"""

import os
import pickle
import numpy as np

from core.embedding import embed_texts, embed_query
from core.llm_interface import call_llm
from core.model_map import MODEL_MAP
from data.airbnb_data import airbnb_database

EMBEDDING_CACHE = os.path.join("data", "airbnb_embeddings.pkl")

SYSTEM_PROMPT = (
    "You are a helpful Airbnb assistant answering questions about Airbnb features, "
    "policies and help-center content. Only answer based on the provided relevant FAQ context. "
    "If you don't know the answer, say: \"Sorry, I don't know the answer to that question. Please consult Airbnb help center.\""
)


# ── Vector store helpers ─────────────────────────────────────────────────────

def _save_cache(path: str, obj: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load_cache(path: str) -> dict | None:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def build_vector_store(corpus: dict[str, str], cache_path: str = EMBEDDING_CACHE) -> dict[str, np.ndarray]:
    """Build or load cached embeddings for every FAQ question in the corpus."""
    keys = list(corpus.keys())
    cache = _load_cache(cache_path)
    if cache and set(cache.keys()) == set(keys):
        return {k: np.array(v, dtype=np.float32) for k, v in cache.items()}

    print(f"Building embeddings for {len(keys)} FAQ entries...")
    vectors = embed_texts(keys)
    store = {keys[i]: vectors[i] for i in range(len(keys))}
    _save_cache(cache_path, {k: v.tolist() for k, v in store.items()})
    return store


# ── Retrieval ────────────────────────────────────────────────────────────────

def retrieve_top_k(
    query_embedding: np.ndarray,
    vector_store: dict[str, np.ndarray],
    k: int = 5,
    min_similarity: float = 0.5,
) -> list[tuple[str, float]]:
    """Return the top-k FAQ keys by cosine similarity above a threshold."""
    keys = list(vector_store.keys())
    matrix = np.vstack([vector_store[k] for k in keys])
    similarities = matrix.dot(query_embedding)

    valid_idx = np.where(similarities >= min_similarity)[0]
    if valid_idx.size == 0:
        return []

    k_eff = min(k, valid_idx.size)
    top_unsorted = valid_idx[np.argpartition(similarities[valid_idx], -k_eff)[-k_eff:]]
    top_sorted = top_unsorted[np.argsort(similarities[top_unsorted])[::-1]]
    return [(keys[i], float(similarities[i])) for i in top_sorted]


# ── RAG pipeline ─────────────────────────────────────────────────────────────

def rag_query(
    query: str,
    vector_store: dict[str, np.ndarray],
    model: str = "gpt-4o-mini",
    k: int = 5,
    min_similarity: float = 0.5,
) -> str:
    """Run the full RAG pipeline: embed query → retrieve → generate answer."""
    query_embedding = embed_query(query)
    hits = retrieve_top_k(query_embedding, vector_store, k=k, min_similarity=min_similarity)

    if not hits:
        return "Sorry, I don't know the answer to that question. Please consult the Airbnb help center."

    print(f"\nRetrieved {len(hits)} relevant FAQ entries:")
    for i, (faq_key, score) in enumerate(hits, 1):
        print(f"  {i}. {faq_key} (similarity={score:.4f})")

    context_blocks = []
    for faq_key, _ in hits:
        answer = airbnb_database.get(faq_key, "")
        if answer:
            context_blocks.append(f"Q: {faq_key}\nA: {answer}")

    if not context_blocks:
        return "Sorry, I don't know the answer to that question. Please consult the Airbnb help center."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": "Relevant FAQs:\n\n" + "\n\n---\n\n".join(context_blocks)},
        {"role": "user", "content": query},
    ]

    return call_llm(messages, model=model)


# ── CLI entry point ──────────────────────────────────────────────────────────

def select_model() -> str:
    """Prompt the user to pick a model from the registry."""
    keys = list(MODEL_MAP.keys())
    print("\nAvailable models:")
    for i, key in enumerate(keys, 1):
        info = MODEL_MAP[key]
        print(f"  {i}. {key}  ({info['model_name']})")

    while True:
        choice = input("Choose a model by number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(keys):
            return keys[int(choice) - 1]
        print("Invalid choice. Try again.")


def main():
    vector_store = build_vector_store(airbnb_database)

    query = input("\nEnter your question: ").strip()
    if not query:
        print("No query provided.")
        return

    model = select_model()
    print(f"\nUsing model: {model}")

    answer = rag_query(query, vector_store, model=model)
    print(f"\n{'─' * 60}\n{answer}\n{'─' * 60}")


if __name__ == "__main__":
    main()
