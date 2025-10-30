import os

from dotenv import load_dotenv
load_dotenv()

from groq import Groq

import time


def fetch_chat_completion(query, model="llama-3.3-70b-versatile", attempt=1) -> str:
    client = Groq(
        api_key=os.getenv("GROQ_API_KEY"),
    )
    if attempt > 5:
        raise Exception("Max retries exceeded for fetch_chat_completion")

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": query,
                }
            ],
            model=model,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        # Check if it's a rate limit error or any other error
        if "rate limit" in str(e).lower() or "429" in str(e):
            print(f"Attempt {attempt}: Rate limit hit, waiting 10 seconds...")
        else:
            print(f"Attempt {attempt}: API call failed with error: {e}, waiting 10 seconds...")
        
        time.sleep(10)
        
        return fetch_chat_completion(query, model=model, attempt=attempt + 1)