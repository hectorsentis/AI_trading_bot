"""Multi-family model support: every family builds, predicts 3-col probabilities, and round-trips."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import joblib

from train import instantiate_classifier
from config import MODEL_FAMILY_PARAMS
from modeling_utils import predict_class_probabilities


def _xy(n=500, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n), "symbol_code": 0})
    y = pd.Series(np.select([X.f1 < -0.4, X.f1 > 0.4], [0, 2], default=1))
    return X, y


@pytest.mark.parametrize("family", ["lgbm", "xgb", "catboost", "rf", "et", "lr"])
def test_family_trains_and_predicts_canonical_3col(family):
    if family in ("xgb", "catboost"):
        pytest.importorskip({"xgb": "xgboost", "catboost": "catboost"}[family])
    X, y = _xy()
    model = instantiate_classifier(family, MODEL_FAMILY_PARAMS[family])
    model.fit(X, y)
    probas = predict_class_probabilities({"model": model}, X.iloc[:5])
    assert probas.shape == (5, 3)
    assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-6)


def test_artifact_roundtrip_predicts(tmp_path):
    X, y = _xy()
    model = instantiate_classifier("rf", MODEL_FAMILY_PARAMS["rf"])
    model.fit(X, y)
    artifact = {"model": model, "model_family": "rf", "feature_columns": ["f1", "f2", "symbol_code"]}
    path = tmp_path / "m.joblib"
    joblib.dump(artifact, path)
    loaded = joblib.load(path)
    probas = predict_class_probabilities(loaded, X.iloc[:3])
    assert probas.shape == (3, 3)


def test_unknown_family_raises():
    with pytest.raises(ValueError):
        instantiate_classifier("not_a_family", {})
