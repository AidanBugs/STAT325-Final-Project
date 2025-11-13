import json
from AI.LLM_Setup import fetch_chat_completion
from AI.LLM_Setup import create_ollama_client
import io
import pandas as pd
import asyncio

def load_resumes():
    with open('data/cleaned_resumes.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def predict_prestige(model=None, local=True, client=None) -> pd.DataFrame:
    if client is None:
        client = create_ollama_client(local=local)
    resumes = load_resumes()
    results = pd.DataFrame()
    batch_size = 5
    for i in range(0, len(resumes), batch_size):
        batch = resumes[i:i + batch_size]
        names = [resume['personal_info']['name'] +": "+ resume['education'][0]['institution']["name"]+" ("+resume["education"][0]["institution"]["location"] + ")" for resume in batch]
        print(names)
        # Create the prompt for the LLM
        prompt = f'''For each institution in the following list, predict their likely prestige level (High/Medium/Low/Unknown). Format the response as CSV. Do not include any explanations or other information. DO NOT use semicolons, use commas as separators. Each prediction should be only from the options provided. 
        
        DO NOT leave an answer as multiple choices. DO NOT leave ethnicity as "Unknown". You MUST provide a single answer for each name.

Example format: 

        Name,Institution,Prestige
        John Doe,Illinois Institute of Technology,Medium
        Kevin Diggs,Boston University,High
        Jane Kim,College of the Canyons,Low
        
        Names to analyze:
        {('\n'.join(names))}'''
        
        try:
            response = fetch_chat_completion(query=str(prompt), model=model, local=local, client=client)
            predictions = pd.read_csv(io.StringIO(response), sep=',')
            results = pd.concat([results, predictions], ignore_index=True)
            print(f"Processed {len(results)} names so far...")
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {str(e)}")
    
    return results


async def predict_prestige_concurrent(model=None, local=True, client=None, max_concurrent=5) -> pd.DataFrame:
    if client is None:
        client = create_ollama_client(local=local)
    resumes = load_resumes()
    results = pd.DataFrame(columns=['name', 'prestige'])
    sepharate = asyncio.Semaphore(max_concurrent) 

    print("Starting concurrent prestige prediction...")

    async def process_batch_resume(batch):
        async with sepharate:
            names = [resume['personal_info']['name'] +"|"+ resume['education'][0]['institution']["name"]+"|"+resume["education"][0]["institution"]["location"] for resume in batch]
            # Create the prompt for the LLM
            prompt = f'''For each institution in the following list, predict their likely prestige level (High/Medium/Low/Unknown). Format the response as CSV. Do not include any explanations or other information. Please use commas as separators. Each prediction should be only from the options provided. Do NOT add a header row.
        
            Input format: 
            
            Name|Institution|Location
            John Doe|Illinois Institute of Technology|Chicago, IL
            Kevin Diggs|Boston University|Boston, MA
            Jane Kim|College of the Canyons|Los Angeles, CA

    Example format: 

            John Doe,Illinois Institute of Technology,Medium
            Kevin Diggs,Boston University,High
            Jane Kim,College of the Canyons,Low
        
            Names to analyze:
            {('\n'.join(names))}'''
        
            try:
                response = await fetch_chat_completion(query=str(prompt), model=model, local=local, client=client)
                predictions = pd.read_csv(io.StringIO(response), sep=',', header=None)
                return predictions
            except Exception as e:
                print(f"Error processing batch starting at index: {str(e)}")
    
    batch_size = 5

    tasks = [process_batch_resume(resumes[i:i + batch_size]) for i in range(0, len(resumes), batch_size)]
    results = pd.concat(await asyncio.gather(*tasks), ignore_index=True, axis=0)

    print(results.head())

    results.columns = ['name', 'institution',  'prestige']

    results.drop(columns=['institution'], inplace=True)

    return results

if __name__ == '__main__':
    print("Starting demographic prediction...")
    results = predict_prestige()
    print(f"Completed! Processed {len(results)} resumes.")
