"""Optional data-backed guard against accidental model regressions."""

import os
from pathlib import Path

import pytest

from affectlens.evaluate import evaluate

MODEL = Path(os.getenv("AFFECTLENS_REGRESSION_MODEL", "artifacts/fer-ferplus.onnx"))
DATA = Path(os.getenv("AFFECTLENS_REGRESSION_DATA", "data/fer-2013/test"))


@pytest.mark.skipif(not MODEL.is_file() or not DATA.is_dir(), reason="local model and FER PublicTest data are required")
def test_publictest_model_regression_thresholds():
    report = evaluate(MODEL, DATA, filename_prefix="PublicTest_")
    assert report["accuracy"] >= 0.30
    assert report["macro_f1"] >= 0.25
