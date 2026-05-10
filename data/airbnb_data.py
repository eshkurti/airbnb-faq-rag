"""Load the Airbnb FAQ dataset from JSON into a question-to-answer mapping."""

import json
import os

_DATA_PATH = os.path.join(os.path.dirname(__file__), "airbnb_faq_with_content.json")


def load_airbnb_database() -> dict[str, str]:
    """Load FAQ entries and return a dict mapping title -> content."""
    if not os.path.exists(_DATA_PATH):
        raise FileNotFoundError(f"Airbnb FAQ JSON not found at {_DATA_PATH}")

    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    db = {}
    for item in data:
        title = item.get("title") or item.get("url") or str(item)
        content = item.get("content", "")
        if title and content:
            db[title] = content
    return db


airbnb_database = load_airbnb_database()
