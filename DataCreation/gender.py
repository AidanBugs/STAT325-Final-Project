import json
from AI.LLM_Setup import fetch_chat_completion
from AI.LLM_Setup import create_ollama_client
import io
import pandas as pd
import asyncio

def load_resumes():
    with open('data/cleaned_resumes.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def predict_demographics(model=None, local=True, client=None) -> pd.DataFrame:
    if client is None:
        client = create_ollama_client(local=local)
    resumes = load_resumes()
    results = pd.DataFrame()
    # Process resumes in batches to avoid making too many API calls
    batch_size = 5
    for i in range(0, len(resumes), batch_size):
        batch = resumes[i:i + batch_size]
        names = [resume['personal_info']['name'] for resume in batch]

        print(names)

        # Create the prompt for the LLM
        prompt = f'''For each name in the following list, predict their likely gender (Male/Female/Unknown) and likely racial/ethnic background based only on the name (Caucasian/Hispanic/African American/Middle Eastern/Asian/South Asian). Format the response as CSV. Do not include any explanations or other information. DO NOT use semicolons, use commas as separators. Each prediction should be only from the options provided. 
        
        DO NOT leave an answer as multiple choices. DO NOT leave ethnicity as "Unknown". You MUST provide a single answer for each name.

Example format: 

        Name,Gender,Ethnicity
        John Doe,Male,Caucasian
        Kevin Diggs,Male,African American
        Jane Kim,Female,Asian
        
        Names to analyze:
        {'\n'.join(names)}'''
        
        try:
            response = fetch_chat_completion(query=str(prompt), model=model, local=local)
            predictions = pd.read_csv(io.StringIO(response), sep=',')
            results = pd.concat([results, predictions], ignore_index=True)
            print(f"Processed {len(results)} names so far...")
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {str(e)}")
    
    return results

async def predict_demographics_concurrent(model=None, local=True, client=None, max_concurrent=5) -> pd.DataFrame:
    if client is None:
        client = create_ollama_client(local=local)
    resumes = load_resumes()
    results = pd.DataFrame()
    sepharate = asyncio.Semaphore(max_concurrent)

    print("Starting concurrent demographic prediction...")
    async def process_batch_resume(batch):
        async with sepharate:
            names = [resume['personal_info']['name'] for resume in batch]

            prompt = f'''For each name in the following list, predict their likely gender (Male/Female/Unknown) and likely racial/ethnic background based only on the name (Caucasian/Hispanic/African American/Middle Eastern/Asian/South Asian). Format the response as CSV. Do not include any explanations or other information. DO NOT use semicolons, use commas as separators. Each prediction should be only from the options provided. 
        
            DO NOT leave an answer as multiple choices. DO NOT leave ethnicity as "Unknown". You MUST provide a single answer for each name. Do NOT include a header row.

            Example format: 

            John Doe,Male,Caucasian
            Kevin Diggs,Male,African American
            Jane Kim,Female,Asian
        
            Names to analyze:
            {'\n'.join(names)}'''


            try:
                response = await fetch_chat_completion(query=str(prompt), model=model, local=local)
                predictions = pd.read_csv(io.StringIO(response), sep=',', header=None, names=['name', 'gender', 'ethnicity'])
                return predictions
            except Exception as e:
                print(f"Error processing batch starting at index: {str(e)}")
    
    batch_size = 5
    tasks = [process_batch_resume(resumes[i:i + batch_size]) for i in range(0, len(resumes), batch_size)]

    predictions = await asyncio.gather(*tasks)

    results = pd.concat(predictions, ignore_index=True, axis=0)
    print(results.head())
    results.columns = ['name', 'gender', 'ethnicity']

    return results

if __name__ == '__main__':
    print("Starting demographic prediction...")
    results = predict_demographics()
    print(f"Completed! Processed {len(results)} resumes.")
