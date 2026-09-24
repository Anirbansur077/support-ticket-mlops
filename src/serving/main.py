import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from scipy.sparse import hstack, csr_matrix
from xgboost import XGBClassifier

app = FastAPI(title="Ticket Escalation Prediction API")

# Load artifacts once at startup
tfidf = joblib.load("src/serving/artifacts/tfidf.pkl")
encoder = joblib.load("src/serving/artifacts/encoder.pkl")
mlb = joblib.load("src/serving/artifacts/tag_binarizer.pkl")
top_tags = joblib.load("src/serving/artifacts/top_tags.pkl")

model = XGBClassifier()
model.load_model("src/serving/artifacts/xgb_escalation_model.json")


class TicketRequest(BaseModel):
    subject: str
    body: str
    type: str
    queue: str
    tags: list[str] = []


class PredictionResponse(BaseModel):
    escalation_risk: float
    escalated: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(ticket: TicketRequest):
    text = ticket.subject + " " + ticket.body
    X_text = tfidf.transform([text])

    cat_df = pd.DataFrame([{"type": ticket.type, "queue": ticket.queue}])
    X_cat = encoder.transform(cat_df)

    tag_list = [t for t in ticket.tags if t in top_tags]
    X_tags = mlb.transform([tag_list])

    X = hstack([X_text, X_cat, csr_matrix(X_tags)])

    prob = model.predict_proba(X)[0][1]
    return PredictionResponse(escalation_risk=float(prob), escalated=bool(prob >= 0.5))