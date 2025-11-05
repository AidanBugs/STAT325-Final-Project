import json
from pathlib import Path
from typing import Any
from DataCreation.gender import predict_demographics
import pandas as pd
import numpy as np


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
        return value.strip() != "" and value.strip().lower() != "n/a" and value.strip().lower() != "unknown" and value.strip().lower() != "not provided" 
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
    return True


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


def remove_missing_name_records(records: list) -> tuple:
    kept = []
    removed = 0
    for rec in records:
        pi = rec.get('personal_info') if isinstance(rec, dict) else None
        if not isinstance(pi, dict):
            removed += 1
            continue
        name = pi.get('name')
        if name is None:
            removed += 1
            continue
        # treat empty/whitespace-only names as missing
        if isinstance(name, str) and (name.strip() == '' or "Newcomer" in name or "Developer"  in name or "Engineer" in name or "Scientist" in name):
            removed += 1
            continue
        kept.append(rec)
    return kept, removed

def remove_missing_school(records: list) -> tuple:
    kept = []
    removed = 0
    for rec in records:
        ed = rec.get('education')[0] if isinstance(rec, dict) else None
        if not isinstance(ed, dict):
            removed += 1
            continue
        inst = ed.get('institution') if isinstance(ed, dict) else None
        if not isinstance(inst, dict):
            removed += 1
            continue
        name = inst.get('name') + ": " + inst.get('location') if isinstance(inst, dict) else None
        if name is None:
            removed += 1
            continue
        # treat empty/whitespace-only names as missing
        if isinstance(name, str) and (name.strip() == ''):
            removed += 1
            continue
        kept.append(rec)
    return kept, removed



def main():
    np.random.seed(42)
    base = Path(__file__).resolve().parent
    src = base / "data\\resumes.jsonl"
    if not src.exists():
        print(f"resumes.jsonl not found at {src}")
        return

    records = list(load_jsonl(src))
    total = len(records)
    print(total)
    cleaned = [r for r in records if not record_is_empty(r)]

    # Remove dupes
    cleaned = [json.loads(j) for j in {json.dumps(d) for d in cleaned}]

    cleaned, removed_missing_name = remove_missing_name_records(cleaned)

    if removed_missing_name:
        print(f"Removed {removed_missing_name} records missing personal_info.name")

    cleaned, removed_missing_school = remove_missing_school(cleaned)

    if removed_missing_school:
        print(f"Removed {removed_missing_school} records missing institution.name")

    # Need to still find GPA's and years of experience



    kept = len(cleaned)

    with open("data\\cleaned_resumes.json", "w", encoding="utf-8") as json_file:
        json_file.write("")
        json.dump(cleaned, json_file, indent=2)


    print("Cleaned resumes:", kept)


    a = input("Do you want to proceed to resume scoring? (y/n): ")
    if a.lower() == 'y':
        print("Starting resume scoring...")
        results = []# Call your resume scoring function here
        print(f"Completed! Processed {len(results)} resumes.")

        filename = "data\\scored_resumes.json"
        f = open(filename, "w+")
        f.close()
        pd.DataFrame(results).to_csv(base / "data\\scored_resumes.json", index=False)

    a = input("Do you want to proceed to demographic prediction? (y/n): ")
    if a.lower() == 'y':
        print("Starting demographic prediction...")
        results = predict_demographics()
        print(f"Completed! Processed {len(results)} resumes.")

        filename = "data\\predicted_demographics.csv"
        f = open(filename, "w+")
        f.close()
        pd.DataFrame(results).to_csv(base / "data\\predicted_demographics.csv", index=False)
    
    a = input("Do you want to proceed to experience and education extraction? (y/n): ")
    if a.lower() == 'y':
        print("Starting experience and education extraction...")
        results = [] # Call your experience and education extraction function here
        print(f"Completed! Processed {len(results)} resumes.")

        filename = "data\\experience_education_extraction.csv"
        f = open(filename, "w+")
        f.close()
        pd.DataFrame(results).to_csv(base / "data\\experience_education_extraction.csv", index=False)

    print("Joining csv data...")
    results = join_csv_data()
    print(f"Completed! Processed {len(results)} resumes.")

    filename = "data\\joined_resumes.csv"
    f = open(filename, "w+")
    f.close()
    pd.DataFrame(results).to_csv(base / "data\\joined_resumes.csv", index=False)

def join_csv_data() -> pd.DataFrame:
    base = Path(__file__).resolve().parent
    scored = pd.read_csv(base / "data\\scored_resumes.json")
    demographics = pd.read_csv(base / "data\\predicted_demographics.csv")
    experience_education = pd.read_csv(base / "data\\experience_education_extraction.csv")

    joined = pd.DataFrame()
    joined = pd.concat([scored, demographics, experience_education], axis=1).reindex(scored.index)

    return joined


if __name__ == "__main__":
    main()
