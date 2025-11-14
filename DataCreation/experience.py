import json
from AI.LLM_Setup import fetch_chat_completion
from AI.LLM_Setup import create_ollama_client
from DataCreation.job_description import load_job_description
import pandas as pd
import asyncio


def load_resumes():
    with open('data/cleaned_resumes.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def get_experience() -> pd.DataFrame:
    """Calculate total years of experience from experience entries in resumes.
    
    Returns a DataFrame with columns 'name' and 'years_experience'.
    """
    resumes = load_resumes()
    results = []

    for resume in resumes:
        try:
            name = resume['personal_info']['name']
            total_years = 0
            
            if 'experience' in resume and isinstance(resume['experience'], list):
                for exp in resume['experience']:
                    if isinstance(exp, dict) and 'dates' in exp:
                        dates = exp['dates']
                        if isinstance(dates, dict) and 'start' in dates and 'end' in dates:
                            try:
                                start_year = float(dates['start'].split('-')[0])
                                end_year = float(dates['end'].split('-')[0])
                                if start_year and end_year and start_year <= end_year:  # validate years
                                    total_years += end_year - start_year
                                start_month = float(dates['start'].split('-')[1])
                                end_month = float(dates['end'].split('-')[1])
                                if start_month and end_month and start_month <= end_month:  # validate months
                                    total_years += (end_month - start_month) / 12.0
                            except (ValueError, TypeError):
                                # Skip entries with invalid year formats
                                continue
            
            results.append({
                'name': name,
                'years_experience': round(total_years, 1)  # round to 1 decimal place
            })
        except Exception as e:
            print(f"Error processing resume for {resume.get('personal_info', {}).get('name', 'Unknown')}: {e}")
    
    return pd.DataFrame(results)

async def score_experience_concurrent(model=None, local=True, max_concurrent=5):
    resumes = load_resumes()
    results = pd.DataFrame(columns=['name', 'score'])
    semaphore = asyncio.Semaphore(max_concurrent)  # Limit concurrent requests

    client = create_ollama_client(local=local)
    
    print("Starting concurrent experience scoring...")
    async def process_single_resume(resume, count=0):
        async with semaphore:
            name = resume['personal_info']['name']
            
            prompt = f'''[{load_job_description()}]

            On a scale of 1 to 100 (only provide a single score), are the below work experiences a good fit for the above job description for a postion at SOFTWARE COMPANY. Provide only the score as an integer. Do not include any explanations or other information. INCLUDING EXTRA INFORMATION WILL BREAK THE CSV FORMAT AND WILL CAUSE ERROR DO NOT DEVIATE FROM THE EXAMPLE FORMAT. PLEASE PLEASE PLEASE DO NOT INCLUDE ```` OR ANY EXTRA CHARACTERS

            Experiences: {resume.get("experience")}

            Example Output format:
            
            Overall Experience Score: 85'''
            
            try:
                response = await fetch_chat_completion(query=str(prompt), model=model, local=local, client=client)
                score = int(response.split(":")[-1].split("\\")[0].strip())
                if count > 0:
                    print(f"Fixed Error {name}")
                return {'name': name, 'experience_score': score}
            except Exception as e:
                if count > 4:
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
