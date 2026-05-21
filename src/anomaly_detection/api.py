"""FastAPI real-time anomaly scoring endpoint."""
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from detector import AnomalyDetector

app = FastAPI(title="Network Anomaly Detection API", version="1.0.0")

try:
    detector = AnomalyDetector.load()
except FileNotFoundError:
    detector = None  # Model not yet trained


class FlowFeatures(BaseModel):
    features: list[float]  # Numeric flow feature vector (e.g., 78 CICIDS features)


class PredictionResult(BaseModel):
    prediction: str  # "normal" or "anomaly"
    anomaly_score: float


@app.post("/predict", response_model=PredictionResult)
async def predict(payload: FlowFeatures) -> PredictionResult:
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first.")
    X = np.array(payload.features).reshape(1, -1)
    pred = detector.predict(X)[0]
    score = float(detector.score_samples(X)[0])
    return PredictionResult(
        prediction="normal" if pred == 1 else "anomaly",
        anomaly_score=round(score, 6),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": detector is not None}
