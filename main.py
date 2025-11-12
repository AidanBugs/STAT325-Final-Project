import json
from pathlib import Path
from typing import Any
from DataCreation.gender import predict_demographics_concurrent
import asyncio
from DataCreation.resume_scorer import score_resumes_concurrent
from DataCreation.prestige import predict_prestige_concurrent
from DataCreation.experience import get_experience
from DataCreation.ollama_utils import select_models
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
        name = inst.get('name') if isinstance(inst, dict) else None
        if name is None:
            removed += 1
            continue
        # treat empty/whitespace-only names as missing
        if isinstance(name, str) and (name.strip() == ''):
            removed += 1
            continue
        location = inst.get('location') if isinstance(inst, dict) else None
        if location is None:
            removed += 1
            continue
        if isinstance(location, str) and (location.strip() == ''):
            removed +=1
            continue
        ach = ed.get('achievements') if isinstance(inst, dict) else None
        if ach is None:
            removed +=1
            continue
        if isinstance(ach, str) and (ach.strip() == ''):
            removed +=1
            continue
        gpa = ach.get('gpa') if isinstance(ed, dict) else None
        if gpa is None:
            removed +=1
            continue
        if isinstance(gpa, str) and (gpa.strip() == ''):
            removed +=1
            continue
        kept.append(rec)

    return kept, removed



async def main():
    np.random.seed(42)
    base = Path(__file__).resolve().parent
    src = base / "data\\resumes.jsonl"
    if not src.exists():
        print(f"resumes.jsonl not found at {src}")
        return

    records = list(load_jsonl(src))
    total = len(records)
    print(f"Total records: {total}")
    cleaned = [r for r in records if not record_is_empty(r)]

    # Remove dupes
    cleaned = [json.loads(j) for j in {json.dumps(d) for d in cleaned}]
    print(f"After removing duplicates: {len(cleaned)}")

    cleaned, removed_missing_name = remove_missing_name_records(cleaned)
    print(f"Removed {removed_missing_name} records missing names")

    cleaned, removed_missing_school = remove_missing_school(cleaned)
    print(f"Removed {removed_missing_school} records missing school info")

    kept = len(cleaned)
    print(f"Final cleaned records: {kept}")

    

    # Create subset for testing
    a = input("Do you want to create a small test subset? (y/n): ")
    if a.lower() == 'y':
        print("\nCreating small test subset...")
        subset = np.random.choice(cleaned, size=10, replace=False)
        with open("data\\cleaned_resumes.json", "w", encoding="utf-8") as json_file:
            json_file.write("")
            json.dump(list(subset), json_file, indent=2)
        print("Subset created successfully.")
    else:
        print("Skipping subset creation.")
        with open("data\\cleaned_resumes.json", "w", encoding="utf-8") as json_file:
            json_file.write("")
            json.dump(cleaned, json_file, indent=2)


    # Resume scoring with multiple models
    a = input("Do you want to proceed to resume scoring? (y/n): ")
    if a.lower() == 'y':
        print("\nStarting resume scoring...")
        
        # Let user select which models to use
        selected_models = select_models()
        if not selected_models:
            print("No models selected. Skipping resume scoring.")
        else:
            print(f"\nScoring resumes using models: {', '.join(selected_models)}")
            # Score resumes with each selected model
            for model in selected_models:
                print(f"\nProcessing with model: {model}")
                
                filename = f"data\\scored_resumes_{model}.csv"
                output_path = base / filename
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # results = score(model=model, client=client, local=True)
                results = await score_resumes_concurrent(model=model, local=True)

                print(results.head())
                # Ensure results is a DataFrame with a name column
                if not isinstance(results, pd.DataFrame):
                    results = pd.DataFrame(results)
                if 'name' not in results.columns and 'personal_info' in cleaned[0]:
                    results['name'] = [r['personal_info']['name'] for r in cleaned]
                demographics = await predict_demographics_concurrent(model=model, local=True)

                print(demographics.head())
                # results = results.join(await predict_demographics_concurrent(model=model, local=True), how='left')
                
                prestige = await predict_prestige_concurrent(model=model, local=True)
                print(prestige.head())
                # results = results.join(await predict_prestige_concurrent(model=model, local=True), how='left')
                
                print(get_experience().head())
                # results = results.join(get_experience(), how='left')
                
                results.to_csv(output_path, index=False)
                print(f"Saved scores for {model} to {filename}")

if __name__ == "__main__":
    asyncio.run(main())
