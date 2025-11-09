import json
from AI.LLM_Setup import fetch_chat_completion
import io
import pandas as pd

def load_resumes():
    with open('data/cleaned_resumes.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def predict_prestige(model=None, local=True) -> pd.DataFrame:
    resumes = load_resumes()
    results = pd.DataFrame()
    # Process resumes in batches to avoid making too many API calls
    batch_size = 5
    for i in range(0, 20, batch_size):
        batch = resumes[i:i + batch_size]
        names = [resume['personal_info']['name'] +": "+ resume['education']['institution']["name"]+" ("+resume["education"]["institution"]["location"] + ")" for resume in batch]

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
            response = fetch_chat_completion(query=str(prompt), model=model, local=local)
            predictions = pd.read_csv(io.StringIO(response), sep=',')
            results = pd.concat([results, predictions], ignore_index=True)
            print(f"Processed {len(results)} names so far...")
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {str(e)}")
    
    return results

if __name__ == '__main__':
    print("Starting demographic prediction...")
    results = predict_prestige()
    print(f"Completed! Processed {len(results)} resumes.")
