import json
from AI.LLM_Setup import fetch_chat_completion
import pandas as pd


def load_resumes():
    with open('data/cleaned_resumes.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def score(model=None, local=True) -> pd.DataFrame:
    resumes = load_resumes()
    results = pd.DataFrame()

    for idx, resume in enumerate(resumes):
        name = resume['personal_info']["name"]
        print(f"Scoring resume {idx + 1}/{len(resumes)}: {name}")

        prompt = f'''Score the following resume on a scale of 1 to 100 based on if the candidate is a good fit for this software company. Provide only the score as an integer. Return only the integer score without any additional text.
        Resume: {resume}'''

        try:
            response = fetch_chat_completion(query=str(prompt), model=model, local=local)
            score = int(response)
            results = pd.concat([results, pd.DataFrame({'name': [name], 'score': [score]})], ignore_index=True)
        except Exception as e:
            print(f"Error scoring resume {idx + 1}: {e}")

    return results