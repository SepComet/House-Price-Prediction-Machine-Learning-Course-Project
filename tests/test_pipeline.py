from __future__ import annotations

import pandas as pd

from house_price_ml.models import ModelResult, RegressorModel, build_default_models
from house_price_ml.pipeline import evaluate_model
from house_price_ml.reporting import build_results_frame


class ConstantRegressor(RegressorModel):
    name = "ConstantRegressor"

    def fit_predict(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
    ) -> pd.Series:
        return pd.Series([1.0] * len(X_test), index=X_test.index)


class MissingRegressor(RegressorModel):
    name = "MissingRegressor"

    def fit_predict(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
    ) -> pd.Series:
        raise NotImplementedError("MissingRegressor is not implemented yet.")


def test_evaluate_model_returns_metrics_for_implemented_model() -> None:
    X_train = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    y_train = pd.Series([1.0, 1.0, 1.0], name="target")
    X_test = pd.DataFrame({"feature": [4.0, 5.0]})
    y_test = pd.Series([1.0, 1.0], name="target")

    result = evaluate_model(ConstantRegressor(), X_train, y_train, X_test, y_test)

    assert isinstance(result, ModelResult)
    assert result.status == "completed"
    assert result.mae == 0.0
    assert result.mse == 0.0
    assert result.rmse == 0.0
    assert result.r2 == 1.0


def test_evaluate_model_skips_unimplemented_model() -> None:
    X_train = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    y_train = pd.Series([1.0, 1.0, 1.0], name="target")
    X_test = pd.DataFrame({"feature": [4.0, 5.0]})
    y_test = pd.Series([1.0, 1.0], name="target")

    result = evaluate_model(MissingRegressor(), X_train, y_train, X_test, y_test)

    assert result.status == "skipped"
    assert result.note == "MissingRegressor is not implemented yet."


def test_default_models_return_predictions() -> None:
    X_train = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})
    y_train = pd.Series([1.0, 2.0, 3.0, 4.0], name="target")
    X_test = pd.DataFrame({"feature": [5.0, 6.0]})

    for model in build_default_models():
        predictions = model.fit_predict(X_train, y_train, X_test)
        assert len(predictions) == len(X_test)
        assert list(predictions.index) == list(X_test.index)


def test_build_results_frame_sorts_completed_models_by_rmse() -> None:
    results = [
        ModelResult(model_name="B", status="completed", mae=0.3, mse=0.09, rmse=0.3, r2=0.7),
        ModelResult(model_name="Skipped", status="skipped", note="not ready"),
        ModelResult(model_name="A", status="completed", mae=0.2, mse=0.04, rmse=0.2, r2=0.8),
    ]

    frame = build_results_frame(results)

    assert list(frame["model_name"]) == ["A", "B", "Skipped"]
    assert list(frame["rank"].iloc[:2]) == [1, 2]
    assert pd.isna(frame["rank"].iloc[2])
