import json
from AI.LLM_Setup import fetch_chat_completion
from AI.LLM_Setup import create_ollama_client
import pandas as pd
import asyncio


def load_resumes():
    with open('data/cleaned_resumes.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def score(model=None, local=True, client=None) -> pd.DataFrame:
    if client is None:
        client = create_ollama_client(local=local)
    resumes = load_resumes()
    results = pd.DataFrame()

    for idx, resume in enumerate(resumes):
        name = resume['personal_info']["name"]
        print(f"Scoring resume {idx + 1}/{len(resumes)}: {name}")

        prompt = f'''Score the following resume on a scale of 1 to 100 based on if the candidate is a good fit for this software company. Provide only the score as an integer. Do not include any explanations or other information.

        Example format:

        John Doe: 85

        Resume: {resume}'''

        try:
            response = fetch_chat_completion(query=str(prompt), model=model, local=local, client=client)
            score = int(response.split(":")[-1].strip())
            results = pd.concat([results, pd.DataFrame({'name': [name], 'score': [score]})], ignore_index=True)
        except Exception as e:
            print(f"Error scoring resume {idx + 1}: {e}")
    


    return results

async def score_resumes_concurrent(model=None, local=True, max_concurrent=5):
    resumes = load_resumes()
    results = pd.DataFrame(columns=['name', 'score'])
    semaphore = asyncio.Semaphore(max_concurrent)  # Limit concurrent requests

    client = create_ollama_client(local=local)
    
    print("Starting concurrent resume scoring...")
    async def process_single_resume(resume, count=0):
        if count > 2:
            print(f"Maximum retry attempts reached for resume {resume['personal_info']['name']}. Skipping...")
            return None
        async with semaphore:
            name = resume['personal_info']['name']
            
            prompt = f'''Score the following resume on a scale of 1 to 100 based on if the candidate is a good fit for this software company. Provide only the score as an integer. Do not include any explanations or other information.

            Example format:

            John Doe: 85

            Resume: {resume}'''
            
            try:
                response = await fetch_chat_completion(query=str(prompt), model=model, local=local, client=client)
                score = int(response.split(":")[-1].strip())
                return {'name': name, 'score': score}
            except Exception as e:
                print(f"Error scoring resume {name}: {e}")
                return process_single_resume(resume, count=count+1)
    
    # Process all resumes concurrently
    tasks = [process_single_resume(resume) for resume in resumes]
    resume_results = await asyncio.gather(*tasks)
    
    # Filter out None results and create DataFrame
    valid_results = [r for r in resume_results if r is not None]
    results = pd.DataFrame(valid_results)
    
    return results