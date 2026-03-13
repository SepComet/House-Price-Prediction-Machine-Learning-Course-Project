from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path


REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "sklearn",
]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cache_root = project_root / ".cache"
    mpl_config_dir = cache_root / "matplotlib"
    sklearn_data_dir = cache_root / "scikit_learn_data"

    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    sklearn_data_dir.mkdir(parents=True, exist_ok=True)

    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

    for module_name in REQUIRED_MODULES:
        import_module(module_name)
        print(f"[ok] import {module_name}")

    from sklearn.datasets import fetch_california_housing

    housing = fetch_california_housing(data_home=sklearn_data_dir)
    print(f"[ok] dataset rows={housing.data.shape[0]} cols={housing.data.shape[1]}")
    print(f"[ok] target name={housing.target_names[0]}")


if __name__ == "__main__":
    main()
