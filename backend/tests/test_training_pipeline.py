import pandas as pd
import pytest

from app.ml.pipelines.training import train_and_compare


def test_classification_pipeline_handles_categorical_features_and_string_labels():
    frame = pd.DataFrame(
        {
            "age": list(range(20, 60)),
            "income": [30000 + index * 1100 for index in range(40)],
            "segment": ["basic" if index % 2 else "premium" for index in range(40)],
            "target": ["yes" if index % 2 else "no" for index in range(40)],
        }
    )

    result = train_and_compare(frame, "target", "classification", tune=False)

    assert set(result["results"]) == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
    }
    assert result["best_model"] in result["results"]
    assert "f1" in result["results"][result["best_model"]]["metrics"]
    assert result["results"][result["best_model"]]["pipeline"].predict(frame.drop(columns="target").head(1))[0] in {"yes", "no"}


def test_regression_pipeline_returns_regression_metrics():
    frame = pd.DataFrame(
        {
            "size": list(range(40)),
            "zone": ["north" if index % 3 else "south" for index in range(40)],
            "price": [100000 + index * 3500 + (index % 3) * 2000 for index in range(40)],
        }
    )

    result = train_and_compare(frame, "price", "regression", tune=False)

    assert result["best_model"] in result["results"]
    assert {"rmse", "r2"}.issubset(result["results"][result["best_model"]]["metrics"])


def test_classification_requires_multiple_sufficiently_populated_classes():
    frame = pd.DataFrame({"feature": range(10), "target": ["only"] * 10})

    with pytest.raises(ValueError, match="at least two target classes"):
        train_and_compare(frame, "target", "classification", tune=False)
