import json
from AI.LLM_Setup import fetch_chat_completion
import io
import pandas as pd

def load_resumes():
    with open('data/cleaned_resumes.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def predict_demographics():
    resumes = load_resumes()
    results = pd.DataFrame()
    # Process resumes in batches to avoid making too many API calls
    batch_size = 40
    for i in range(0, 50, batch_size):
        batch = resumes[i:i + batch_size]
        names = [resume['personal_info']['name'] for resume in batch]

        # Create the prompt for the LLM
        prompt = f'''For each name in the following list, predict their likely gender (Male/Female) and likely racial/ethnic background based only on the name. Format the response as CSV. Do not include any explanations or other information, only the CSV data.

Example format: 

        Name;Gender;Ethnicity
        John Doe;Male;Caucasian
        Jane Kim;Female;Asian
        
        Names to analyze:
        {('\n'.join(names))}'''
        
        try:
            response = fetch_chat_completion(prompt)
            predictions = pd.read_csv(io.StringIO(response), sep=';')
            results = pd.concat([results, predictions], ignore_index=True)
            print(f"Processed {len(results)} names so far...")
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {str(e)}")
    
    return results

if __name__ == '__main__':
    print("Starting demographic prediction...")
    results = predict_demographics()
    print(f"Completed! Processed {len(results)} resumes.")
