# Airbnb FAQ — Retrieval-Augmented Generation Chatbot

A RAG chatbot that answers questions about Airbnb by retrieving relevant FAQ entries and grounding LLM responses in retrieved context. Built as a university project for the **Practical Studies Industry 4.0** course at Hochschule Hof.

## How It Works

```
User Query
    │
    ▼
┌──────────────┐     ┌───────────────────┐
│  Embed query │────▶│  Vector similarity │
│  (text-      │     │  search (top-k,    │
│  embedding-  │     │  cosine similarity │
│  3-small)    │     │  with threshold)   │
└──────────────┘     └────────┬──────────┘
                              │
                              ▼
                     ┌────────────────┐
                     │ Retrieved FAQ  │
                     │ entries (k=5)  │
                     └────────┬───────┘
                              │
                              ▼
                     ┌────────────────┐
                     │ LLM generates  │
                     │ grounded answer│
                     └────────────────┘
```

**473 Airbnb help-center FAQ entries** are embedded into a vector store. When a user asks a question, the system retrieves the most semantically similar entries (using cosine similarity with a configurable threshold), injects them as context, and generates a grounded answer. This prevents hallucination by constraining the LLM to only use retrieved source material.

## Project Structure

```
airbnb-faq-rag/
├── rag_chat.py                 # Main RAG pipeline (embed → retrieve → generate)
├── context_extension_chat.py   # Baseline: full-context approach (no retrieval)
├── core/
│   ├── embedding.py            # Text embedding via OpenRouter
│   ├── llm_interface.py        # Chat completions via OpenRouter
│   └── model_map.py            # Model registry
├── data/
│   ├── airbnb_data.py          # FAQ JSON loader
│   └── airbnb_faq_with_content.json  # 473 FAQ entries
├── requirements.txt
└── .env.example
```

## Two Approaches Compared

| Approach | Script | Method | Limitation |
|---|---|---|---|
| **RAG** | `rag_chat.py` | Embed + retrieve top-k + LLM | Requires embedding step |
| **Full Context** | `context_extension_chat.py` | Stuff entire FAQ into prompt | Hits context window limits, higher cost |

The full-context baseline demonstrates *why* RAG is needed: stuffing all 473 FAQ entries into a single prompt is expensive and can exceed context window limits on smaller models.

## Setup

```bash
git clone https://github.com/eshkurti/airbnb-faq-rag.git
cd airbnb-faq-rag

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt

cp .env .env
# Edit .env and add your OpenRouter API key
```

## Usage

**RAG chatbot:**
```bash
python rag_chat.py
```

On first run, the system builds embeddings for all 473 FAQ entries and caches them locally. Subsequent runs load from cache.

**Full-context baseline:**
```bash
python context_extension_chat.py
```

Both scripts let you choose from multiple LLM providers (GPT-4o, Gemini, LLaMA, Claude) via OpenRouter.

## Tech Stack

- **Embeddings**: OpenAI `text-embedding-3-small` (via OpenRouter)
- **LLMs**: Multi-model support via OpenRouter (GPT-4o, Gemini 2.5 Flash, LLaMA 4, Claude Sonnet 4)
- **Vector Search**: NumPy-based cosine similarity with top-k retrieval and minimum similarity threshold
- **Data**: 473 Airbnb help-center FAQ entries (scraped, JSON format)

## Configuration

Key parameters in `rag_chat.py`:

- `k=5` — number of FAQ entries to retrieve per query
- `min_similarity=0.5` — cosine similarity threshold (filters out irrelevant results)
- Embedding cache is stored at `data/airbnb_embeddings.pkl`
