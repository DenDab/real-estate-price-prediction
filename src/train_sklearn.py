"""Baseline scikit-learn models for House Prices, compared via 5-fold CV.

Target is log1p(SalePrice) throughout (matches the Kaggle RMSE-on-log-price
metric). For linear models with a regularization strength (Ridge, Lasso),
alpha is first selected with an inner GridSearchCV, then the resulting
pipeline is scored with fresh 5-fold out-of-fold predictions so every model
in the comparison table is evaluated the same way.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline

from preprocessing import TARGET_COL, build_full_pipeline, load_train_test

RANDOM_STATE = 42
N_FOLDS = 5


def make_model_pipeline(X_sample: pd.DataFrame, model) -> Pipeline:
    """Feature engineering + ColumnTransformer (from preprocessing.py) + model."""
    preprocess_steps = build_full_pipeline(X_sample).steps
    return Pipeline(preprocess_steps + [("model", model)])


def build_model_specs(X_sample: pd.DataFrame):
    """Return {name: (pipeline, param_grid_or_None)} for every baseline model."""
    return {
        "LinearRegression": (
            make_model_pipeline(X_sample, LinearRegression()),
            None,
        ),
        "Ridge": (
            make_model_pipeline(X_sample, Ridge(random_state=RANDOM_STATE)),
            {"model__alpha": [0.1, 1, 5, 10, 20, 30, 50, 75, 100]},
        ),
        "Lasso": (
            make_model_pipeline(X_sample, Lasso(max_iter=50000, random_state=RANDOM_STATE)),
            {"model__alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]},
        ),
        "RandomForest": (
            make_model_pipeline(X_sample, RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
            {"model__n_estimators": [200, 400], "model__max_depth": [None, 10, 20]},
        ),
        "HistGradientBoosting": (
            make_model_pipeline(X_sample, HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
            {
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [None, 6],
                "model__max_iter": [200, 400],
            },
        ),
    }


def tune_alpha(pipeline: Pipeline, param_grid: dict, X, y, cv) -> Pipeline:
    """Select the best hyperparameters via GridSearchCV, return an unfit pipeline set to them."""
    search = GridSearchCV(
        pipeline, param_grid, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1,
    )
    search.fit(X, y)
    print(f"    best params: {search.best_params_}")
    return search.best_estimator_


def evaluate_model(pipeline: Pipeline, X, y, cv):
    """5-fold out-of-fold RMSE (log scale) + MAE/R2 (dollar scale)."""
    y_pred_log = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=-1)

    rmse_log = np.sqrt(mean_squared_error(y, y_pred_log))
    price_true = np.expm1(y)
    price_pred = np.expm1(y_pred_log)
    mae_price = mean_absolute_error(price_true, price_pred)
    r2_price = r2_score(price_true, price_pred)

    return {"rmse_log": rmse_log, "mae_price": mae_price, "r2_price": r2_price}, y_pred_log


def run_comparison(X: pd.DataFrame, y: pd.Series):
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    specs = build_model_specs(X)

    rows = []
    predictions = {}
    for name, (pipeline, param_grid) in specs.items():
        print(f"[{name}]")
        if param_grid is not None:
            pipeline = tune_alpha(pipeline, param_grid, X, y, cv)
        metrics, y_pred_log = evaluate_model(pipeline, X, y, cv)
        print(f"    rmse_log={metrics['rmse_log']:.4f}  mae_price=${metrics['mae_price']:,.0f}  r2_price={metrics['r2_price']:.4f}")
        rows.append({"model": name, **metrics})
        predictions[name] = y_pred_log

    results = pd.DataFrame(rows).set_index("model").sort_values("rmse_log")
    return results, predictions


if __name__ == "__main__":
    train_df, _ = load_train_test("data")
    X = train_df.drop(columns=[TARGET_COL])
    y = np.log1p(train_df[TARGET_COL])

    results, _ = run_comparison(X, y)
    print("\n=== Model comparison (sorted by RMSE on log price) ===")
    print(results.to_string(float_format=lambda v: f"{v:,.4f}"))
