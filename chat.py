import os 
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load variables from .env if you have one
load_dotenv() 

# 2. Define your key
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# 3. Pass the local variable 'GROQ_API_KEY' instead of 'os.environ.get(...)'
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY  # Changed this line
)
while True:
    user_prompt = input("Prompt: ")
    system_prompt = "Your answer should not limit in one sentence."

# 4. Use the Groq-compatible Responses API
    response = client.responses.create(
        model="llama-3.3-70b-versatile",
        instructions=system_prompt,
        input=user_prompt
    )

    print(response.output_text)

