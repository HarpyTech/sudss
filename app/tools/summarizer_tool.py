import os
from langchain.tools import Tool
from vertexai.language_models import TextGenerationModel

GEMINI_PROJECT = os.getenv("GCP_PROJECT")
GEMINI_LOCATION = os.getenv("GCP_REGION", "us-central1")

def gemini_summarize(text: str) -> str:
    model = TextGenerationModel.from_pretrained("gemini-1.5-flash-preview")
    response = model.predict(text, max_output_tokens=512)
    return response.text

summarizer_tool = Tool(
    name="Summarizer Tool",
    func=gemini_summarize,
    description="Use Gemini or Gemma to summarize BioGPT and PubMedGPT outputs."
)
