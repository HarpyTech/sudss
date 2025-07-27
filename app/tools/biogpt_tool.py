import os
from langchain.tools import Tool
import requests

BIOGPT_API_URL = os.getenv("BIOGPT_API_URL")
BIOGPT_API_KEY = os.getenv("BIOGPT_API_KEY")

def call_biogpt(text: str) -> str:
    headers = {
        "Authorization": f"Bearer {BIOGPT_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(BIOGPT_API_URL, json={"input": text}, headers=headers)
    response.raise_for_status()
    return response.json().get("output", "BioGPT response unavailable.")

biogpt_tool = Tool(
    name="BioGPT Tool",
    func=call_biogpt,
    description="Use this tool for biological & medical language-based diagnosis using BioGPT."
)
