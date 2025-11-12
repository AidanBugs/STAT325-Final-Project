import json
from AI.LLM_Setup import fetch_chat_completion
import pandas as pd


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