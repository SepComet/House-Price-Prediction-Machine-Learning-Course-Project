from __future__ import annotations

from dataclasses import asdict, dataclass
from urllib.error import URLError

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from house_price_ml.config import RAW_DATA_DIR, SKLEARN_DATA_DIR


@dataclass(slots=True)
class DatasetSummary:
    rows: int
    feature_count: int
    target_name: str
    missing_values: int
    target_mean: float
    target_std: float

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


class DatasetLoadError(RuntimeError):
    """Raised when the California Housing dataset cannot be loaded."""


def load_california_housing() -> tuple[pd.DataFrame, pd.Series]:
    SKLEARN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        dataset = fetch_california_housing(as_frame=True, data_home=SKLEARN_DATA_DIR)
    except (OSError, URLError) as exc:
        raise DatasetLoadError(
            "Failed to load California Housing dataset. "
            f"Expected local cache under '{SKLEARN_DATA_DIR}', or network access for first download."
        ) from exc

    features = dataset.data.copy()
    target = dataset.target.copy()
    target.name = dataset.target_names[0]
    return features, target


def summarize_dataset(features: pd.DataFrame, target: pd.Series) -> DatasetSummary:
    return DatasetSummary(
        rows=len(features),
        feature_count=features.shape[1],
        target_name=target.name or "target",
        missing_values=int(features.isna().sum().sum() + target.isna().sum()),
        target_mean=float(target.mean()),
        target_std=float(target.std()),
    )


def save_raw_dataset(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = features.copy()
    dataset[target.name or "target"] = target
    dataset.to_csv(RAW_DATA_DIR / "california_housing.csv", index=False)
    return dataset


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )
