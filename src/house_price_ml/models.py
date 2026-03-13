from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


@dataclass(slots=True)
class ModelResult:
    model_name: str
    status: str
    mae: float | None = None
    mse: float | None = None
    rmse: float | None = None
    r2: float | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, str | float | None]:
        return asdict(self)


class RegressorModel(ABC):
    name: str

    @abstractmethod
    def fit_predict(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
    ) -> pd.Series:
        raise NotImplementedError


class SklearnRegressorModel(RegressorModel):
    def __init__(self, name: str, estimator: object) -> None:
        self.name = name
        self.estimator = estimator

    def fit_predict(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
    ) -> pd.Series:
        self.estimator.fit(X_train, y_train)
        predictions = self.estimator.predict(X_test)
        return pd.Series(predictions, index=X_test.index, name=self.name)


def build_default_models() -> list[RegressorModel]:
    return [
        SklearnRegressorModel(
            "LinearRegression",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("regressor", LinearRegression()),
                ]
            ),
        ),
        SklearnRegressorModel(
            "DecisionTreeRegressor",
            DecisionTreeRegressor(random_state=42),
        ),
        SklearnRegressorModel(
            "RandomForestRegressor",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=1,
            ),
        ),
    ]
