"""In-memory aggregate confidence, latency, and prediction-distribution monitoring."""

from collections import Counter, deque

import numpy as np

from .labels import LABELS


class PredictionMonitor:
    def __init__(self, baseline: dict[str, float] | None = None, window_size: int = 256, drift_threshold: float = 0.2):
        self.baseline = np.asarray([float((baseline or {}).get(label, 0.0)) for label in LABELS], dtype=np.float64)
        if self.baseline.sum() > 0:
            self.baseline /= self.baseline.sum()
        self.window: deque[tuple[str, float, float]] = deque(maxlen=window_size)
        self.total_count = 0
        self.drift_threshold = drift_threshold

    def record(self, probabilities: dict[str, float], latency_ms: float):
        if not probabilities:
            return
        label = max(probabilities, key=probabilities.get)
        confidence = float(probabilities[label])
        self.window.append((label, confidence, float(latency_ms)))
        self.total_count += 1

    def snapshot(self) -> dict:
        if not self.window:
            return {
                "total_count": self.total_count,
                "recent_count": 0,
                "mean_latency_ms": None,
                "p95_latency_ms": None,
                "confidence_histogram": [0] * 10,
                "predicted_class_distribution": {},
                "prediction_drift_psi": None,
                "drift_threshold": self.drift_threshold,
                "drift_alert": False,
                "note": "In-memory aggregates only; a drift alert compares predicted-label frequency with the supplied benchmark baseline and is not a fairness or accuracy test.",
            }
        labels = [row[0] for row in self.window]
        confidence = np.asarray([row[1] for row in self.window])
        latency = np.asarray([row[2] for row in self.window])
        counts = Counter(labels)
        distribution = {label: counts[label] / len(labels) for label in LABELS}
        psi = None
        if self.baseline.sum() > 0 and len(labels) >= 50:
            observed = np.asarray([distribution[label] for label in LABELS])
            expected = np.clip(self.baseline, 1e-5, 1)
            observed = np.clip(observed, 1e-5, 1)
            psi = float(np.sum((observed - expected) * np.log(observed / expected)))
        return {
            "total_count": self.total_count,
            "recent_count": len(self.window),
            "mean_latency_ms": float(latency.mean()),
            "p95_latency_ms": float(np.percentile(latency, 95)),
            "confidence_histogram": np.histogram(confidence, bins=np.linspace(0, 1, 11))[0].tolist(),
            "predicted_class_distribution": distribution,
            "prediction_drift_psi": psi,
            "drift_threshold": self.drift_threshold,
            "drift_alert": bool(psi is not None and psi >= self.drift_threshold),
            "note": "In-memory aggregates only; a drift alert compares predicted-label frequency with the supplied benchmark baseline and is not a fairness or accuracy test.",
        }
