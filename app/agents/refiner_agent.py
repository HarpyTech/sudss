from openai import OpenAI

from agents.retrieval_agent import RetrievalAgent

client = OpenAI()

class RefinerAgent:
    def __init__(self):
        self.retriever = RetrievalAgent()

    def refine(self, draft_report: str, query: str):
        similar_cases = self.retriever.retrieve(query, k=3)
        context = "\n".join([f"{c['type']}: {c['text']}" for c in similar_cases])

        prompt = f"""
        Given draft report: {draft_report}
        Context from similar cases: {context}
        Refine the report to improve accuracy and clarity.
        """
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"You are a medical refiner agent."},
                      {"role":"user","content":prompt}]
        )
        return resp.choices[0].message.content
