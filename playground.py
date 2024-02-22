from openai import OpenAI
from pathlib import Path

prompt = Path("prompts/qa.prompt").read_text()
prompt = prompt.format(
    question="Who is Francesco", context="Francesco is a Machine Learning Engineer"
)
print(prompt)
client = OpenAI(
    base_url="http://localhost:11434/v1/",
    # required but ignored
    api_key="mistral",
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": prompt,
        }
    ],
    model="mistral",
)
print(chat_completion)
