"""Preprocessing pipeline for the Ames Housing (House Prices) dataset."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Columns where NaN means "feature absent", filled with "None" (categorical)
# or 0 (numeric) rather than imputed.
NONE_CATEGORICAL_COLS = [
    "Alley", "MasVnrType", "BsmtQual", "BsmtCond", "BsmtExposure",
    "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType",
    "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature",
]
ZERO_NUMERIC_COLS = [
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "GarageYrBlt", "GarageCars", "GarageArea",
]

# Ordinal quality scales: worst -> best. Missing/absent already filled as "None".
QUALITY_SCALE = ["None", "Po", "Fa", "TA", "Gd", "Ex"]
QUALITY_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]
BSMT_EXPOSURE_SCALE = ["None", "No", "Mn", "Av", "Gd"]
BSMT_FINTYPE_SCALE = ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"]
GARAGE_FINISH_SCALE = ["None", "Unf", "RFn", "Fin"]

ORDINAL_COLS = {
    **{col: QUALITY_SCALE for col in QUALITY_COLS},
    "BsmtExposure": BSMT_EXPOSURE_SCALE,
    "BsmtFinType1": BSMT_FINTYPE_SCALE,
    "BsmtFinType2": BSMT_FINTYPE_SCALE,
    "GarageFinish": GARAGE_FINISH_SCALE,
}

# Nominal categoricals (no inherent order) -> one-hot encoded.
NOMINAL_COLS = [
    "MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities",
    "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2",
    "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st",
    "Exterior2nd", "MasVnrType", "Foundation", "Heating", "CentralAir",
    "Electrical", "Functional", "GarageType", "PavedDrive", "MiscFeature",
    "SaleType", "SaleCondition", "Fence",
]

ID_COL = "Id"
TARGET_COL = "SalePrice"


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the two well-known GrLivArea/SalePrice outliers (train only)."""
    mask = (df["GrLivArea"] > 4000) & (df[TARGET_COL] < 300000)
    return df.loc[~mask].reset_index(drop=True)


def fill_structural_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill features whose NaN means 'absent', stateless (no fold-dependent stats)."""
    df = df.copy()

    for col in NONE_CATEGORICAL_COLS:
        df[col] = df[col].fillna("None")
    for col in ZERO_NUMERIC_COLS:
        df[col] = df[col].fillna(0)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that summarize related raw columns."""
    df = df.copy()

    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBath"] = (
        df["FullBath"] + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    )
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["YearsSinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)

    return df


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Structural fills + engineered features as a pipeline-compatible step.

    LotFrontage has no natural "0 means none" reading, so its per-neighborhood
    (and global) medians are learned in `fit` and reused in `transform`, so a
    CV test fold is always filled with training-fold statistics rather than
    recomputing medians from itself.
    """

    def fit(self, X, y=None):
        filled = fill_structural_missing(X)
        self.neighborhood_frontage_medians_ = filled.groupby("Neighborhood")["LotFrontage"].median()
        self.global_frontage_median_ = filled["LotFrontage"].median()
        return self

    def transform(self, X):
        X = fill_structural_missing(X)
        X["LotFrontage"] = X["LotFrontage"].fillna(
            X["Neighborhood"].map(self.neighborhood_frontage_medians_)
        )
        X["LotFrontage"] = X["LotFrontage"].fillna(self.global_frontage_median_)
        X = engineer_features(X)
        return X


def build_preprocessor(numeric_cols, ordinal_cols, nominal_cols) -> ColumnTransformer:
    """ColumnTransformer: scale numerics, ordinal-encode quality scales,
    one-hot encode nominal categoricals. Fit only on train folds to avoid leakage.
    """
    ordinal_categories = [ORDINAL_COLS[col] for col in ordinal_cols]

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    ordinal_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="None")),
        ("encode", OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])
    nominal_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("ord", ordinal_pipeline, ordinal_cols),
        ("nom", nominal_pipeline, nominal_cols),
    ])


def build_full_pipeline(df: pd.DataFrame) -> Pipeline:
    """Build the full sklearn Pipeline: feature engineering -> ColumnTransformer.

    Column lists are derived from `df` *after* feature engineering, so pass a
    representative sample (e.g. the training set) to get consistent numeric/
    ordinal/nominal splits.
    """
    engineered = FeatureBuilder().fit(df).transform(df)
    drop_cols = {ID_COL, TARGET_COL}
    ordinal_cols = [c for c in ORDINAL_COLS if c in engineered.columns]
    nominal_cols = [c for c in NOMINAL_COLS if c in engineered.columns]
    numeric_cols = [
        c for c in engineered.select_dtypes(include=[np.number]).columns
        if c not in drop_cols and c not in ordinal_cols and c not in nominal_cols
    ]

    preprocessor = build_preprocessor(numeric_cols, ordinal_cols, nominal_cols)

    return Pipeline([
        ("features", FeatureBuilder()),
        ("preprocess", preprocessor),
    ])


def load_train_test(data_dir: str = "../data"):
    train = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    train = remove_outliers(train)
    return train, test


if __name__ == "__main__":
    train_df, test_df = load_train_test("data")

    X = train_df.drop(columns=[TARGET_COL])
    y = np.log1p(train_df[TARGET_COL])

    pipeline = build_full_pipeline(X)
    X_transformed = pipeline.fit_transform(X)

    print(f"Train shape after preprocessing: {X_transformed.shape}")
    print(f"Target (log SalePrice) stats: mean={y.mean():.3f}, std={y.std():.3f}")
