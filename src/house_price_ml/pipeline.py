from __future__ import annotations

from math import sqrt

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from house_price_ml.data import (
    load_california_housing,
    save_raw_dataset,
    split_dataset,
    summarize_dataset,
)
from house_price_ml.models import ModelResult, RegressorModel, build_default_models
from house_price_ml.reporting import export_results


def evaluate_model(
    model: RegressorModel,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> ModelResult:
    try:
        predictions = model.fit_predict(X_train, y_train, X_test)
    except NotImplementedError as exc:
        return ModelResult(
            model_name=model.name,
            status="skipped",
            note=str(exc),
        )

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = sqrt(mse)
    r2 = r2_score(y_test, predictions)
    return ModelResult(
        model_name=model.name,
        status="completed",
        mae=float(mae),
        mse=float(mse),
        rmse=float(rmse),
        r2=float(r2),
    )


def run_pipeline(models: list[RegressorModel] | None = None) -> dict[str, object]:
    features, target = load_california_housing()
    save_raw_dataset(features, target)

    dataset_summary = summarize_dataset(features, target)
    X_train, X_test, y_train, y_test = split_dataset(features, target)

    model_list = models if models is not None else build_default_models()
    results = [
        evaluate_model(model, X_train, y_train, X_test, y_test)
        for model in model_list
    ]

    output_paths = export_results(dataset_summary, results)
    return {
        "dataset_summary": dataset_summary,
        "results": results,
        "output_paths": output_paths,
    }
