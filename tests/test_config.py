from house_price_ml.config import DATA_DIR, FIGURES_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR


def test_project_paths_are_relative_to_data_dir() -> None:
    assert RAW_DATA_DIR.parent == DATA_DIR
    assert PROCESSED_DATA_DIR.parent == DATA_DIR
    assert FIGURES_DIR.parent.name == "reports"
