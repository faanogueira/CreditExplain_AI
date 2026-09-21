from uuid import uuid4
from fastapi import FastAPI
from pydantic import BaseModel, Field

from credit_explain.data.credit_domain import good_explanation
from credit_explain.risk.demo_engine import score_application

app = FastAPI(title="CreditExplain AI", version="0.1.0")


class ApplicationFeatures(BaseModel):
    application_id: str
    debt_to_income: float = Field(ge=0, le=1.5)
    bureau_score: int = Field(ge=300, le=850)
    utilization: float = Field(ge=0, le=1.5)
    delinquencies_12m: int = Field(ge=0, le=20)
    employment_tenure_months: int = Field(ge=0, le=600)
    requested_amount: float = Field(gt=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/explain")
def explain(application: ApplicationFeatures):
    features = application.model_dump()
    scored = score_application(features)
    context = {**features, **scored}
    return {
        "application_id": application.application_id,
        "risk": scored,
        "explanation": good_explanation(context),
        "trace_id": str(uuid4()),
        "model_version": "demo_policy_renderer",
        "note": "Replace demo renderer with the fine tuned adapter for model serving.",
    }
