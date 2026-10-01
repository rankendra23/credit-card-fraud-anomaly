from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.fraud_detection import AnomalyFraudPipeline, FEATURE_COLUMNS

app = FastAPI(
    title="Credit Card Fraud Detection API",
    version="1.0.0",
)

MODEL_PATH = Path("artifacts/best_model.joblib")

pipeline = AnomalyFraudPipeline.load(MODEL_PATH)
MODEL_NAME = "One-Class SVM"


class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
    }


@app.post("/predict")
def predict(transaction: Transaction):
    row = pd.DataFrame(
        [transaction.model_dump()]
    )[FEATURE_COLUMNS]

    score = float(
        pipeline.score(row, MODEL_NAME)[0]
    )

    prediction = int(
        pipeline.predict(row, MODEL_NAME)[0]
    )

    return {
        "prediction": "suspicious" if prediction == 1 else "normal",
        "is_anomaly": prediction,
        "anomaly_score": score,
        "model": MODEL_NAME,
    }
