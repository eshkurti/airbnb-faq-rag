"""
Baseline: Full-context LLM chat (no retrieval).

Passes the entire FAQ dataset into the LLM context window as a comparison
to the RAG approach. Demonstrates context window limitations and the
motivation for retrieval-augmented generation.
"""

from core.llm_interface import call_llm
from core.model_map import MODEL_MAP
from data.airbnb_data import airbnb_database

SYSTEM_PROMPT = (
    "You are a helpful Airbnb assistant answering questions about Airbnb features, "
    "policies and help-center content. Only answer based on the provided relevant FAQ context. "
    "If you don't know the answer, say: \"Sorry, I don't know the answer to that question. Please consult Airbnb help center.\""
)


def build_full_context_prompt(database: dict[str, str], question: str) -> list[dict]:
    """Build a prompt that includes the entire FAQ database in the context."""
    context_lines = [f"Q: {key}\nA: {answer}" for key, answer in database.items()]
    context_text = "\n\n".join(context_lines)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Complete FAQ database:\n{context_text}"},
        {"role": "user", "content": question},
    ]


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
    query = input("\nEnter your question: ").strip()
    if not query:
        print("No query provided.")
        return

    model = select_model()
    print(f"\nUsing model: {model}")
    print(f"Sending entire FAQ database ({len(airbnb_database)} entries) as context...\n")

    messages = build_full_context_prompt(airbnb_database, query)
    answer = call_llm(messages, model=model)
    print(f"\n{'─' * 60}\n{answer}\n{'─' * 60}")


if __name__ == "__main__":
    main()
