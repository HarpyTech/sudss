from langchain.tools import Tool

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

model_name = "microsoft/MediPhi-PubMed"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}


def pubmedgpt_model(text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": """Extract medical keywords from this operative notes
              focus on anatomical, pathological, or procedural vocabulary.""",
        },
        {"role": "user", "content": text},
    ]
    output = pipe(messages, **generation_args)

    return output[0]["generated_text"]


description = "Tool for retrieving clinical insights from PubMed-based GPT analysis."
pubmedgpt_tool = Tool(
    name="PubMedGPT Tool",
    func=pubmedgpt_model,
    description=description,
)
