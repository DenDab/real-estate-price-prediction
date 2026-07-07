"""Helpers for feature-importance interpretation of the fitted pipelines."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Plain-language labels for the features that tend to dominate every
# importance ranking, used only for the "non-technical audience" charts.
HUMAN_LABELS = {
    "OverallQual": "Overall quality (1-10)",
    "GrLivArea": "Living area",
    "TotalSF": "Total house area",
    "Neighborhood": "Neighborhood",
    "TotalBsmtSF": "Basement area",
    "1stFlrSF": "1st floor area",
    "GarageCars": "Garage capacity (cars)",
    "GarageArea": "Garage area",
    "TotalBath": "Number of bathrooms",
    "YearBuilt": "Year built",
    "HouseAge": "House age",
    "KitchenQual": "Kitchen quality",
    "ExterQual": "Exterior quality",
    "FullBath": "Number of full bathrooms",
    "Fireplaces": "Number of fireplaces",
    "2ndFlrSF": "2nd floor area",
    "OverallCond": "Overall condition",
    "CentralAir": "Central air conditioning",
    "YearRemodAdd": "Year of last remodel",
    "MSZoning": "Zoning classification",
    "LotArea": "Lot area",
}


def get_output_feature_names(preprocessor):
    """Return (display_names, base_names) for a fitted ColumnTransformer
    built by preprocessing.build_preprocessor.

    display_names: one entry per output column (one-hot dummies expanded),
    base_names: the original raw/engineered column each output column came
    from, so one-hot dummies of the same categorical collapse back together
    (e.g. all `Neighborhood_*` columns share base name "Neighborhood").
    """
    num_cols = list(preprocessor.transformers_[0][2])
    ord_cols = list(preprocessor.transformers_[1][2])
    nom_cols = list(preprocessor.transformers_[2][2])

    ohe = preprocessor.named_transformers_["nom"].named_steps["encode"]
    nom_display = list(ohe.get_feature_names_out(nom_cols))
    nom_base = [col for col, cats in zip(nom_cols, ohe.categories_) for _ in cats]

    display_names = num_cols + ord_cols + nom_display
    base_names = num_cols + ord_cols + nom_base
    return display_names, base_names


def aggregate_by_base_feature(values, base_names, agg="sum_abs"):
    """Collapse one-hot-expanded importances/coefficients back to their
    original column (e.g. sum |coefficient| across all Neighborhood dummies).
    """
    df = pd.DataFrame({"base": base_names, "value": values})
    if agg == "sum_abs":
        out = df.assign(value=df["value"].abs()).groupby("base")["value"].sum()
    elif agg == "sum":
        out = df.groupby("base")["value"].sum()
    else:
        raise ValueError(f"Unknown agg: {agg}")
    return out.sort_values(ascending=False)


def raw_permutation_importance(pipeline: Pipeline, X_test: pd.DataFrame, y_test, **kwargs):
    """Permutation importance computed on the ORIGINAL raw columns (via the
    full feature-engineering + preprocessing + model pipeline). This is more
    interpretable than transformed-feature importance since categoricals
    like Neighborhood come back as a single column instead of many dummies.
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(pipeline, X_test, y_test, **kwargs)
    importances = pd.Series(result.importances_mean, index=X_test.columns)
    return importances.sort_values(ascending=False)
