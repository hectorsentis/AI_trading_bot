"""Phase B: calibrated-probability helper.

Verifies the live prediction path uses the artifact's calibrator when present, falls back to the
raw model otherwise, and always returns columns in canonical [SHORT, FLAT, LONG] order even when
the estimator reports a different `classes_` ordering.
"""
from __future__ import annotations

import numpy as np

from modeling_utils import predict_class_probabilities, CLASS_SHORT, CLASS_FLAT, CLASS_LONG


class _Estimator:
    def __init__(self, proba_row, classes):
        self._proba_row = np.asarray(proba_row, dtype=float)
        self.classes_ = list(classes)

    def predict_proba(self, X):
        return np.tile(self._proba_row, (len(X), 1))


def test_uses_calibrator_when_present():
    model = _Estimator([0.6, 0.3, 0.1], [0, 1, 2])      # raw
    calibrator = _Estimator([0.1, 0.3, 0.6], [0, 1, 2])  # calibrated
    artifact = {"model": model, "calibrator": calibrator}
    probas = predict_class_probabilities(artifact, [[1.0, 2.0]])
    assert probas[0].tolist() == [0.1, 0.3, 0.6]  # calibrator wins


def test_falls_back_to_model_without_calibrator():
    model = _Estimator([0.6, 0.3, 0.1], [0, 1, 2])
    artifact = {"model": model, "calibrator": None}
    probas = predict_class_probabilities(artifact, [[1.0, 2.0]])
    assert probas[0].tolist() == [0.6, 0.3, 0.1]


def test_reorders_to_canonical_class_order():
    # Estimator reports classes as [LONG, FLAT, SHORT]; helper must reorder to [SHORT, FLAT, LONG].
    est = _Estimator([0.7, 0.2, 0.1], [CLASS_LONG, CLASS_FLAT, CLASS_SHORT])
    artifact = {"model": est, "calibrator": None}
    probas = predict_class_probabilities(artifact, [[0.0]])
    # canonical: SHORT=0.1, FLAT=0.2, LONG=0.7
    assert probas[0].tolist() == [0.1, 0.2, 0.7]
