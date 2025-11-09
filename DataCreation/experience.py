import json
from AI.LLM_Setup import fetch_chat_completion
import pandas as pd


def load_resumes():
    with open('data/cleaned_resumes.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def predict_experience(model=None, local=True) -> pd.DataFrame:
    resumes = load_resumes()
    results = pd.DataFrame()

    for idx, resume in enumerate(resumes):
        name = resume['personal_info']["name"]
        print(f"Finding Years of Experience for resume {idx + 1}/{len(resumes)}: {name}")

        prompt = f'''Find the years of relevant experience for the following resume that applies to this software company. Return only the integer years of experience without any additional text.
        Resume: {resume['resume']}'''

        try:
            response = fetch_chat_completion(query=str(prompt), model=model, local=local)
            score = int(response)
            results = pd.concat([results, pd.DataFrame({'name': [name], 'years_relevant_experience': [score]})], ignore_index=True)
        except Exception as e:
            print(f"Error scoring resume {idx + 1}: {e}")

    return results