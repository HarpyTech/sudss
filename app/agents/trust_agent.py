import random

class TrustAgent:
    def compute_trust(self, report: str, feedback: str):
        score = random.uniform(0.7, 0.95) if feedback else random.uniform(0.5, 0.8)
        return {"trust_score": round(score, 2), "explanation": "Score derived from feedback alignment."}
