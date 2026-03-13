from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from house_price_ml.config import PROCESSED_DATA_DIR, REPORTS_DIR
from house_price_ml.data import DatasetSummary
from house_price_ml.models import ModelResult


def build_results_frame(results: list[ModelResult]) -> pd.DataFrame:
    results_frame = pd.DataFrame([result.to_dict() for result in results])
    if results_frame.empty:
        return results_frame

    completed_mask = results_frame["status"] == "completed"
    completed_frame = results_frame.loc[completed_mask].copy()
    skipped_frame = results_frame.loc[~completed_mask].copy()

    if not completed_frame.empty:
        completed_frame = completed_frame.sort_values(
            by=["rmse", "mae", "r2"],
            ascending=[True, True, False],
        ).reset_index(drop=True)
        completed_frame["rank"] = range(1, len(completed_frame) + 1)

    if not skipped_frame.empty:
        skipped_frame["rank"] = pd.NA

    ordered_columns = [
        "rank",
        "model_name",
        "status",
        "mae",
        "mse",
        "rmse",
        "r2",
        "note",
    ]
    combined_frame = pd.concat([completed_frame, skipped_frame], ignore_index=True)
    return combined_frame.reindex(columns=ordered_columns)


def export_results(
    dataset_summary: DatasetSummary,
    results: list[ModelResult],
) -> dict[str, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    summary_payload: dict[str, Any] = {
        "dataset": dataset_summary.to_dict(),
        "models": [asdict(result) for result in results],
    }
    summary_path = REPORTS_DIR / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    results_frame = build_results_frame(results)
    results_csv_path = REPORTS_DIR / "model_results.csv"
    results_frame.to_csv(results_csv_path, index=False)

    processed_summary = pd.DataFrame([dataset_summary.to_dict()])
    processed_summary_path = PROCESSED_DATA_DIR / "dataset_summary.csv"
    processed_summary.to_csv(processed_summary_path, index=False)

    return {
        "summary_json": summary_path,
        "results_csv": results_csv_path,
        "dataset_csv": processed_summary_path,
    }
