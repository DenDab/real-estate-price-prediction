"""PyTorch MLP baseline for House Prices, evaluated the same way as train_sklearn.py.

Reuses the sklearn preprocessing Pipeline (feature engineering + scaling +
one-hot/ordinal encoding) to build the input matrix, then trains a small MLP
per CV fold with early stopping on an inner validation split. Out-of-fold
predictions are collected the same way cross_val_predict does for the
sklearn models, so RMSE/MAE/R2 are directly comparable across the whole
model-comparison table.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from preprocessing import TARGET_COL, build_full_pipeline, load_train_test

RANDOM_STATE = 42
N_FOLDS = 5
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 300
PATIENCE = 40  # epochs without val-loss improvement before early stopping;
# generous because the ~10% validation split is small and per-epoch val loss is noisy
GRAD_CLIP_NORM = 5.0  # a MiscVal outlier (~$15.5k vs mostly $0) produces large
# standardized inputs that destabilize training without clipping

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HousePriceMLP(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one_fold(X_train, y_train, X_val, y_val, n_features, seed):
    torch.manual_seed(seed)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    model = HousePriceMLP(n_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, epoch + 1, best_val_loss


def run_pytorch_cv(X: pd.DataFrame, y: pd.Series):
    """5-fold CV with the same splits as train_sklearn.py's KFold(random_state=42).

    Each training fold is further split 90/10 into fit/early-stopping-val sets.
    """
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_pred_log = np.zeros(len(y))

    for fold, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
        y_train_raw, y_test_raw = y.iloc[train_idx], y.iloc[test_idx]

        preprocessor = build_full_pipeline(X_train_raw)
        X_train_full = preprocessor.fit_transform(X_train_raw)
        X_test = preprocessor.transform(X_test_raw)
        if hasattr(X_train_full, "toarray"):
            X_train_full = X_train_full.toarray()
        if hasattr(X_test, "toarray"):
            X_test = X_test.toarray()

        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train_full, y_train_raw.values, test_size=0.1, random_state=RANDOM_STATE,
        )

        model, n_epochs, best_val_loss = train_one_fold(
            X_fit, y_fit, X_val, y_val, n_features=X_train_full.shape[1], seed=RANDOM_STATE + fold,
        )

        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE)).cpu().numpy()
        y_pred_log[test_idx] = preds
        print(f"  fold {fold}/{N_FOLDS}: stopped after {n_epochs} epochs, best val MSE={best_val_loss:.4f}")

    rmse_log = np.sqrt(mean_squared_error(y, y_pred_log))
    price_true = np.expm1(y)
    price_pred = np.expm1(y_pred_log)
    mae_price = mean_absolute_error(price_true, price_pred)
    r2_price = r2_score(price_true, price_pred)

    metrics = {"rmse_log": rmse_log, "mae_price": mae_price, "r2_price": r2_price}
    return metrics, y_pred_log


if __name__ == "__main__":
    train_df, _ = load_train_test("data")
    X = train_df.drop(columns=[TARGET_COL])
    y = np.log1p(train_df[TARGET_COL])

    print(f"Device: {DEVICE}")
    print("[PyTorch MLP]")
    metrics, _ = run_pytorch_cv(X, y)
    print(f"    rmse_log={metrics['rmse_log']:.4f}  mae_price=${metrics['mae_price']:,.0f}  r2_price={metrics['r2_price']:.4f}")
