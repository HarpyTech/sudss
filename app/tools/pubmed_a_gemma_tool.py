from langchain.tools import Tool
import google.generativeai as genai
from config.variables import GOOGLE_AI_API_KEY
from config.constants import GEMMA3_4B

# Configure Google GenAI SDK
genai.configure(api_key=GOOGLE_AI_API_KEY)

# Initialize the Gemma model
model = genai.GenerativeModel(GEMMA3_4B)


# Define the keyword extraction function
def gemma_pubmed_model(text: str) -> str:
    prompt = (
        "Extract medical keywords from the following operative notes. "
        "Focus on anatomical, pathological, or procedural vocabulary.\n\n"
        f"{text}"
    )
    response = model.generate_content(prompt)
    return response.text.strip()


# Define LangChain Tool
pubmedgpt_tool = Tool(
    name="Gemma PubMed Tool",
    func=gemma_pubmed_model,
    description="Tool using Google AI Studio's Gemma model for clinical keyword extraction.",
)
