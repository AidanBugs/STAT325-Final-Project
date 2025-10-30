import json
from pathlib import Path
from typing import Any


def is_nonempty(value: Any) -> bool:
    """Return True if a value is meaningfully non-empty.

    Treats as empty: None, empty string or whitespace-only string,
    empty list/dict/tuple/set, and containers that only contain empty values.
    Numbers and booleans count as non-empty.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        return value.strip() != "" and value.strip().lower() != "n/a" and value.strip().lower() != "unknown"
    if isinstance(value, (list, tuple, set)):
        for v in value:
            if not is_nonempty(v):
                return False
        return True
    if isinstance(value, dict):
        for v in value.values():
            if not is_nonempty(v):
                return False
        return True
    # Fallback: consider it non-empty
    return True


def record_is_empty(record: Any) -> bool:
    """Return True if the entire record contains no non-empty leaves."""
    if "personal_info" in record:
        return not is_nonempty(record["personal_info"])
    return False


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                # Show which line failed to parse and continue
                print(f"Warning: failed to parse JSON on line {i}: {e}")


def main():
    base = Path(__file__).resolve().parent
    src = base / "resumes.jsonl"
    if not src.exists():
        print(f"resumes.jsonl not found at {src}")
        return

    records = list(load_jsonl(src))
    total = len(records)
    print(total)
    cleaned = [r for r in records if not record_is_empty(r)]
    kept = len(cleaned)

    print(kept)


if __name__ == "__main__":
    main()
