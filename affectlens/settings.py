import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_path: Path | None
    face_cascade_path: Path | None
    confidence_threshold: float
    max_upload_bytes: int
    save_images: bool
    rate_limit_per_minute: int = 120

    @classmethod
    def from_env(cls) -> "Settings":
        model = os.getenv("AFFECTLENS_MODEL_PATH")
        cascade = os.getenv("AFFECTLENS_FACE_CASCADE")
        threshold = float(os.getenv("AFFECTLENS_CONFIDENCE_THRESHOLD", "0.55"))
        max_upload_bytes = int(os.getenv("AFFECTLENS_MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
        save_images = os.getenv("SAVE_IMAGES", "false").strip().casefold() in {"1", "true", "yes", "on"}
        rate_limit = int(os.getenv("AFFECTLENS_RATE_LIMIT_PER_MINUTE", "120"))
        if save_images:
            raise ValueError("SAVE_IMAGES=true is unsupported: uploaded images are never written to disk")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("AFFECTLENS_CONFIDENCE_THRESHOLD must be between 0 and 1")
        if max_upload_bytes <= 0:
            raise ValueError("AFFECTLENS_MAX_UPLOAD_BYTES must be a positive integer")
        if rate_limit <= 0:
            raise ValueError("AFFECTLENS_RATE_LIMIT_PER_MINUTE must be positive")
        return cls(
            model_path=Path(model) if model else None,
            face_cascade_path=Path(cascade) if cascade else None,
            confidence_threshold=threshold,
            max_upload_bytes=max_upload_bytes,
            save_images=save_images,
            rate_limit_per_minute=rate_limit,
        )
