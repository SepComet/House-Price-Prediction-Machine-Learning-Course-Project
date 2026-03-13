from __future__ import annotations

import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from house_price_ml.data import DatasetLoadError
from house_price_ml.pipeline import run_pipeline
from house_price_ml.reporting import build_results_frame


def main() -> None:
    try:
        run_result = run_pipeline()
    except DatasetLoadError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    dataset_summary = run_result["dataset_summary"]
    results = run_result["results"]
    output_paths = run_result["output_paths"]

    print(
        "dataset:",
        f"rows={dataset_summary.rows}",
        f"features={dataset_summary.feature_count}",
        f"target={dataset_summary.target_name}",
    )

    results_frame = build_results_frame(results)
    completed_results = results_frame.loc[results_frame["status"] == "completed"]
    if not completed_results.empty:
        print("ranking:")
        for row in completed_results.itertuples(index=False):
            print(
                f"#{int(row.rank)} {row.model_name} "
                f"rmse={row.rmse:.4f} mae={row.mae:.4f} r2={row.r2:.4f}"
            )

    skipped_results = results_frame.loc[results_frame["status"] != "completed"]
    for row in skipped_results.itertuples(index=False):
        print(f"model={row.model_name} status={row.status} note={row.note}")

    for name, path in output_paths.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
