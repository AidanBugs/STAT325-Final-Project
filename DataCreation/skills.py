import json
from AI.LLM_Setup import fetch_chat_completion
from AI.LLM_Setup import create_ollama_client
from DataCreation.job_description import load_job_description
import pandas as pd
import asyncio


def load_resumes():
    with open('data/cleaned_resumes.json', 'r', encoding='utf-8') as f:
        return json.load(f)

async def score_skills_concurrent(model=None, local=True, max_concurrent=5):
    resumes = load_resumes()
    results = pd.DataFrame(columns=['name', 'score'])
    semaphore = asyncio.Semaphore(max_concurrent)  # Limit concurrent requests

    client = create_ollama_client(local=local)
    
    print("Starting concurrent skill scoring...")
    async def process_single_resume(resume, count=0):
        async with semaphore:
            name = resume['personal_info']['name']
            
            prompt = f'''[{load_job_description()}]

            On a scale of 1 to 100 (only provide a single score), are the below skills a good fit for the above job description for a postion at SOFTWARE COMPANY. Provide only the score as an integer. Do not include any explanations or other information. INCLUDING EXTRA INFORMATION WILL BREAK THE CSV FORMAT AND WILL CAUSE ERROR DO NOT DEVIATE FROM THE EXAMPLE FORMAT. PLEASE PLEASE PLEASE DO NOT INCLUDE ```` OR ANY EXTRA CHARACTERS

            Skills: {resume.get("skills")}

            Example Output format:

            Skill Score: 85'''
            
            try:
                response = await fetch_chat_completion(query=str(prompt), model=model, local=local, client=client)
                score = int(response.split(":")[-1].split("\\")[0].strip())
                if count > 0:
                    print(f"Fixed Error {name}")
                return {'name': name, 'skill_score': score}
            except Exception as e:
                if count > 10:
                    raise ValueError(e)
                print(f"Error scoring resume {name}: {e}")
                return await process_single_resume(resume, count=count+1)
    
    # Process all resumes concurrently
    tasks = [process_single_resume(resume) for resume in resumes]
    resume_results = await asyncio.gather(*tasks)
    
    # Filter out None results and create DataFrame
    valid_results = [r for r in resume_results if r is not None]
    results = pd.DataFrame(valid_results)
    
    return results
