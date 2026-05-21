"""Ensemble anomaly detector: Isolation Forest + One-Class SVM."""
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from pathlib import Path

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


class AnomalyDetector:
    """
    Ensemble of Isolation Forest and One-Class SVM.
    Returns -1 for anomaly, 1 for normal traffic.
    """

    def __init__(self, contamination: float = 0.05):
        self.scaler = StandardScaler()
        self.iforest = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1
        )
        self.ocsvm = OneClassSVM(kernel="rbf", nu=contamination, gamma="scale")

    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        X_scaled = self.scaler.fit_transform(X)
        self.iforest.fit(X_scaled)
        self.ocsvm.fit(X_scaled)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble vote: anomaly only if both models agree."""
        X_scaled = self.scaler.transform(X)
        if_pred = self.iforest.predict(X_scaled)   # -1 or 1
        svm_pred = self.ocsvm.predict(X_scaled)    # -1 or 1
        # Flag as anomaly (-1) only when both agree
        return np.where((if_pred == -1) & (svm_pred == -1), -1, 1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return Isolation Forest anomaly scores (lower = more anomalous)."""
        return self.iforest.score_samples(self.scaler.transform(X))

    def save(self, name: str = "detector") -> None:
        joblib.dump(self, MODEL_DIR / f"{name}.pkl")

    @staticmethod
    def load(name: str = "detector") -> "AnomalyDetector":
        return joblib.load(MODEL_DIR / f"{name}.pkl")
