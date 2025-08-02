from langchain.tools import Tool
import google.generativeai as genai
from config import constants, variables

# Configure API key
if not variables.GOOGLE_AI_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY environment variable")

genai.configure(api_key=variables.GOOGLE_AI_API_KEY)


# Define Gemini summarization function using Google GenAI SDK
def gemini_summarize(text: str) -> str:
    model = genai.GenerativeModel(constants.GEMINI25_FLASH)
    response = model.generate_content(
        f"Summarize the following medical insights:\n{text}"
    )
    return response.text if hasattr(response, "text") else str(response)


# Define LangChain-compatible Tool
summarizer_tool = Tool(
    name="Summarizer Tool",
    func=gemini_summarize,
    description="Use Gemini Pro (via Google GenAI SDK) to summarize outputs from BioGPT and PubMedGPT.",
)
