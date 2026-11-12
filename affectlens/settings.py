from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_path: Path | None
    face_cascade_path: Path | None
    confidence_threshold: float
    max_upload_bytes: int

    @classmethod
    def from_env(cls) -> "Settings":
        model = os.getenv("AFFECTLENS_MODEL_PATH")
        cascade = os.getenv("AFFECTLENS_FACE_CASCADE")
        threshold = float(os.getenv("AFFECTLENS_CONFIDENCE_THRESHOLD", "0.55"))
        max_upload_bytes = int(os.getenv("AFFECTLENS_MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("AFFECTLENS_CONFIDENCE_THRESHOLD must be between 0 and 1")
        if max_upload_bytes <= 0:
            raise ValueError("AFFECTLENS_MAX_UPLOAD_BYTES must be a positive integer")
        return cls(
            model_path=Path(model) if model else None,
            face_cascade_path=Path(cascade) if cascade else None,
            confidence_threshold=threshold,
            max_upload_bytes=max_upload_bytes,
        )
