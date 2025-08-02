from langchain.tools import Tool
from transformers import BioGptTokenizer, BioGptForCausalLM

tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

def call_biogpt(text: str) -> str:
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    
    return output

biogpt_tool = Tool(
    name="BioGPT Tool",
    func=call_biogpt,
    description="Use this tool for biological & medical language-based diagnosis using BioGPT."
)
