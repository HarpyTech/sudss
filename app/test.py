from fastapi import FastAPI
from pydantic import BaseModel
from agents.refiner_agent import RefinerAgent
from agents.feedback_agent import FeedbackAgent
from agents.trust_agent import TrustAgent
from agents.explanation_agent import ExplanationAgent

app = FastAPI()
refiner, feedback, trust, explainer = (
    RefinerAgent(),
    FeedbackAgent(),
    TrustAgent(),
    ExplanationAgent(),
)


class ReportRequest(BaseModel):
    draft_report: str
    query: str


class FeedbackRequest(BaseModel):
    report: str
    corrections: str


@app.post("/refine")
def refine_report(req: ReportRequest):
    return {"refined_report": refiner.refine(req.draft_report, req.query)}


@app.post("/feedback")
def apply_feedback(req: FeedbackRequest):
    corrected = feedback.apply_feedback(req.report, req.corrections)
    score = trust.compute_trust(corrected, req.corrections)
    return {"corrected_report": corrected, "trust": score}


@app.post("/explain")
def explain_report(req: ReportRequest):
    return {"explanation": explainer.explain(req.draft_report)}
