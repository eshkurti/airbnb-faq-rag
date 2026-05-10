# core/model_map.py

MODEL_MAP = {
    # Google Gemini via OpenRouter
    "gemini-2.5-flash": {"provider": "openrouter", "model_name": "google/gemini-2.5-flash"},

    # Meta LLaMA via OpenRouter
    "llama-4-maverick": {"provider": "openrouter", "model_name": "meta-llama/llama-4-maverick"},

    # OpenAI GPT via OpenRouter
    "gpt-4o": {"provider": "openrouter", "model_name": "openai/gpt-4o"},

    # Anthropic Claude via OpenRouter
    "claude-opus-4.1": {"provider": "openrouter", "model_name": "anthropic/claude-opus-4.1"},
}
