import json
from pathlib import Path
from typing import Any
from DataCreation.gender import predict_demographics_concurrent
import asyncio
from DataCreation.resume_scorer import score_resumes_concurrent
from DataCreation.prestige import predict_prestige_concurrent
from DataCreation.experience import get_experience
from DataCreation.experience import score_experience_concurrent
from DataCreation.skills import score_skills_concurrent
from DataCreation.projects import score_projects_concurrent
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


def remove_missing_skills_projects_records(records: list) -> tuple:
    kept = []
    removed = 0
    for rec in records:
        sk = rec.get('skills') if isinstance(rec, dict) else None
        if not isinstance(sk, dict):
            removed += 1
            continue
        pr = rec.get('projects') if isinstance(rec, dict) else None
        if not isinstance(pr, list):
            removed += 1
            continue
        kept.append(rec)
    return kept, removed

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


def clean_resumes(records: list) -> tuple:
    cleaned = [r for r in records if not record_is_empty(r)]

    # Remove dupes
    cleaned = [json.loads(j) for j in {json.dumps(d) for d in cleaned}]
    print(f"After removing duplicates: {len(cleaned)}")

    cleaned, removed_missing_name = remove_missing_name_records(cleaned)
    print(f"Removed {removed_missing_name} records missing names")

    cleaned, removed_missing_school = remove_missing_school(cleaned)
    print(f"Removed {removed_missing_school} records missing school info")

    cleaned, removed_missing_skills = remove_missing_skills_projects_records(cleaned)
    print(f"Removed {removed_missing_skills} records missing skills or project info")

    kept = len(cleaned)
    print(f"Final cleaned records: {kept}")
    return cleaned



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

    record_path = base / "data\\ready_resumes.json"
    if not record_path.exists():
        cleaned = clean_resumes(records)
        print("Writing cleaned resumes to data/ready_resumes.json")

        with open("data\\ready_resumes.json", "w", encoding="utf-8") as json_file:
            json_file.write("")
            json.dump(cleaned, json_file, indent=2)
    else:
        with open(record_path, 'r', encoding='utf-8') as f:
            cleaned = json.load(f)
        print(f"Loaded cleaned resumes from {record_path}, total: {len(cleaned)}")

    sub = input("Do you want to create a smaller subset (progress saved)? (y/n): ")
    if sub.lower() == 'y':
        size = int(input("Enter the size of the subset: "))
    else:
        size = len(cleaned)
    a = 'y'
    if a.lower() == 'y':
        print("\nStarting resume scoring...")
        selected_models = select_models()
        if not selected_models:
            print("No models selected. Skipping resume scoring.")
        else:
            print(f"\nScoring resumes using models: {', '.join(selected_models)}")
            # Score resumes with each selected model
            for model in selected_models:
                print(f"\nProcessing with model: {model}")

                filename = f"data\\{model.replace(':', '_')}_resume_scores.csv"
                output_path = base / filename
                output_path.parent.mkdir(parents=True, exist_ok=True)

                if output_path.exists():
                    current = pd.read_csv(output_path).shape[0]
                else:
                    current = 0
                subset = cleaned[current:current+size]
                
                with open("data\\cleaned_resumes.json", "w", encoding="utf-8") as json_file:
                    json_file.write("")
                    json.dump(list(subset), json_file, indent=2)
                print("Resumes to score saved successfully.")

                max_concurrent = 14
                results = await score_resumes_concurrent(model=model, local=True, max_concurrent=max_concurrent)
                demographics = await predict_demographics_concurrent(model=model, local=True, max_concurrent=max_concurrent)
                prestige = await predict_prestige_concurrent(model=model, local=True, max_concurrent=max_concurrent)
                skills = await score_skills_concurrent(model=model, local=True, max_concurrent=max_concurrent)
                projects = await score_projects_concurrent(model=model, local=True, max_concurrent=max_concurrent)
                exp = await score_experience_concurrent(model=model, local=True, max_concurrent=max_concurrent)
                

                print(f"Merging results...")
                results = results.merge(demographics, how='left', on='name')
                results = results.merge(prestige, how='left', on='name')
                results = results.merge(skills, how='left', on='name')
                results = results.merge(projects, how='left', on='name')
                results = results.merge(exp, how='left', on='name')
                results = results.merge(get_experience(), how='left', on='name')

                if output_path.exists():
                    print(f"Appending to existing file {filename}...")
                    results.to_csv(output_path, mode='a', header=False, index=False)
                else:
                    print(f"Output file {filename} does not exist. Creating a new file...")
                    results.to_csv(output_path, index=False)
                
                print(f"Saved scores for {model} to {filename}")

if __name__ == "__main__":
    asyncio.run(main())
