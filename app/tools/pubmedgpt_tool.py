from langchain.tools import Tool

def mock_pubmedgpt(text: str) -> str:
    return f"PubMedGPT Research Reference: Based on current literature - {text[:200]}"

pubmedgpt_tool = Tool(
    name="PubMedGPT Tool",
    func=mock_pubmedgpt,
    description="Tool for retrieving clinical insights from PubMed-based GPT analysis."
)
