from openai import OpenAI

client = OpenAI()

class ExplanationAgent:
    def explain(self, report: str):
        prompt = f"Explain step by step how this diagnostic impression was derived:\n\n{report}"
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"You are an explanation agent for clinicians."},
                      {"role":"user","content":prompt}]
        )
        return resp.choices[0].message.content
