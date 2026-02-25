
from pathlib import Path

def load_text() -> str:
    text_path = Path("data/text.txt")
    text_data = text_path.read_text(encoding="utf-8")
    print("Loaded text data (%d characters)", len(text_data))
    print("First 100 characters: %s", text_data[:100])
    return text_data