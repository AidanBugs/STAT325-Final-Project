import os

from dotenv import load_dotenv
load_dotenv()

from groq import Groq


def fetch_chat_completion(query, model="llama-3.3-70b-versatile"):
    client = Groq(
        api_key=os.getenv("GROQ_API_KEY"),
    )
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        model=model,
    )
    
    return (chat_completion.choices[0].message.content)
