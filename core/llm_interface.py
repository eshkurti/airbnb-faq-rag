"""LLM interface for chat completions via OpenRouter."""

import os
import requests
from dotenv import load_dotenv
from core.model_map import MODEL_MAP

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_llm(messages: list[dict], model: str = "gpt-4o", max_tokens: int = 400, temperature: float = 0.0) -> str:
    """Send a chat completion request to OpenRouter and return the response text."""
    if model not in MODEL_MAP:
        raise ValueError(f"Unknown model '{model}'. Available: {list(MODEL_MAP.keys())}")

    model_name = MODEL_MAP[model]["model_name"]

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )

    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter API error {response.status_code}: {response.text}")

    return response.json()["choices"][0]["message"]["content"]
