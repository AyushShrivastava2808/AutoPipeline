import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from app.ml.pipelines.preprocessing import build_preprocessor, infer_column_types


class LabelEncodedClassifier(ClassifierMixin, BaseEstimator):
    """Allow all classification estimators, including XGBoost, to accept string labels."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder().fit(y)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, self.label_encoder_.transform(y))
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, X):
        encoded_predictions = self.estimator_.predict(X)
        return self.label_encoder_.inverse_transform(encoded_predictions.astype(int))

CLASSIFICATION_MODELS = {
    "logistic_regression": LabelEncodedClassifier(LogisticRegression(max_iter=1000)),
    "decision_tree": LabelEncodedClassifier(DecisionTreeClassifier(random_state=42)),
    "random_forest": LabelEncodedClassifier(RandomForestClassifier(random_state=42)),
    "xgboost": LabelEncodedClassifier(XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=1)),
}

REGRESSION_MODELS = {
    "linear_regression": LinearRegression(),
    "decision_tree": DecisionTreeRegressor(random_state=42),
    "random_forest": RandomForestRegressor(random_state=42),
    "xgboost": XGBRegressor(random_state=42, n_jobs=1),
}

# Hyperparameter grids, keyed by model name. Params are prefixed with "model__"
# since each estimator sits inside a Pipeline step named "model".
CLASSIFICATION_PARAM_GRIDS = {
    "logistic_regression": {"model__estimator__C": [0.01, 0.1, 1.0, 10.0]},
    "decision_tree": {"model__estimator__max_depth": [3, 5, 10, None], "model__estimator__min_samples_split": [2, 5, 10]},
    "random_forest": {"model__estimator__n_estimators": [100, 200, 300], "model__estimator__max_depth": [5, 10, None]},
    "xgboost": {"model__estimator__n_estimators": [100, 200], "model__estimator__max_depth": [3, 5, 7], "model__estimator__learning_rate": [0.01, 0.1, 0.3]},
}

REGRESSION_PARAM_GRIDS = {
    "linear_regression": {},
    "decision_tree": {"model__max_depth": [3, 5, 10, None], "model__min_samples_split": [2, 5, 10]},
    "random_forest": {"model__n_estimators": [100, 200, 300], "model__max_depth": [5, 10, None]},
    "xgboost": {"model__n_estimators": [100, 200], "model__max_depth": [3, 5, 7], "model__learning_rate": [0.01, 0.1, 0.3]},
}


def train_and_compare(df: pd.DataFrame, target_column: str, task_type: str, tune: bool = True) -> dict:
    df = _validate_training_data(df, target_column, task_type)
    numeric_cols, categorical_cols = infer_column_types(df, target_column)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    X = df.drop(columns=[target_column])
    y = df[target_column]
    stratify = y if task_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    if task_type == "classification":
        cv = min(3, int(y_train.value_counts().min()))
    else:
        cv = min(3, len(y_train))

    models = CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS
    param_grids = CLASSIFICATION_PARAM_GRIDS if task_type == "classification" else REGRESSION_PARAM_GRIDS
    results = {}

    for name, estimator in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        grid = param_grids.get(name, {})
        best_params = {}

        if tune and grid:
            n_candidates = min(10, _grid_size(grid))
            search = RandomizedSearchCV(
                pipeline, param_distributions=grid, n_iter=n_candidates,
                cv=cv, random_state=42, n_jobs=-1,
            )
            search.fit(X_train, y_train)
            pipeline = search.best_estimator_
            best_params = search.best_params_
        else:
            pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)

        if task_type == "classification":
            metrics = {
                "accuracy": accuracy_score(y_test, preds),
                "f1": f1_score(y_test, preds, average="weighted"),
            }
        else:
            metrics = {
                "rmse": mean_squared_error(y_test, preds, squared=False),
                "r2": r2_score(y_test, preds),
            }

        results[name] = {
            "pipeline": pipeline,
            "metrics": metrics,
            "best_params": best_params,
            "feature_importance": _extract_feature_importance(pipeline),
        }

    ranking_key = "f1" if task_type == "classification" else "r2"
    best_name = max(results, key=lambda n: results[n]["metrics"][ranking_key])

    return {"results": results, "best_model": best_name, "feature_columns": list(X.columns)}


def _grid_size(grid: dict) -> int:
    size = 1
    for values in grid.values():
        size *= len(values)
    return size


def _validate_training_data(df: pd.DataFrame, target_column: str, task_type: str) -> pd.DataFrame:
    if task_type not in {"classification", "regression"}:
        raise ValueError("task_type must be 'classification' or 'regression'")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is not present in the dataset")

    prepared = df.dropna(subset=[target_column]).copy()
    if len(prepared) < 10:
        raise ValueError("At least 10 rows with a target value are required for training")
    if len(prepared.columns) < 2:
        raise ValueError("The dataset must contain at least one feature column and one target column")

    if task_type == "classification":
        class_counts = prepared[target_column].value_counts()
        if len(class_counts) < 2:
            raise ValueError("Classification requires at least two target classes")
        if class_counts.min() < 4:
            raise ValueError("Each classification target class needs at least four samples")
    else:
        numeric_target = pd.to_numeric(prepared[target_column], errors="coerce")
        if numeric_target.isna().any():
            raise ValueError("Regression requires a numeric target column")
        prepared[target_column] = numeric_target

    return prepared


def _extract_feature_importance(pipeline: Pipeline, top_n: int = 20) -> dict | None:
    model = pipeline.named_steps["model"]
    if isinstance(model, LabelEncodedClassifier):
        model = model.estimator_
    preprocessor = pipeline.named_steps["preprocessor"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return None

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        values = abs(coef[0]) if coef.ndim > 1 else abs(coef)
    else:
        return None

    ranked = sorted(zip(feature_names, values.tolist()), key=lambda pair: -pair[1])
    return dict(ranked[:top_n])
