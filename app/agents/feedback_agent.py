class FeedbackAgent:
    def apply_feedback(self, report: str, corrections: str):
        return f"Corrected Report:\n{report}\n---Applied Corrections---\n{corrections}"
