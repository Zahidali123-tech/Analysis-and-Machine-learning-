from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from data_utils import load_data

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "Sales"
NUMERIC = ["Customer_Age", "Quantity", "Unit_Price", "Discount_Percentage", "Order_Month", "Order_Day", "Order_Year"]
CATEGORICAL = ["Gender", "City", "Region", "Category", "Sub_Category", "Payment_Method", "Customer_Type"]
FEATURES = NUMERIC + CATEGORICAL

def make_features(df):
    data = df.copy()
    data["Order_Date"] = pd.to_datetime(data["Order_Date"], errors="coerce")
    data["Order_Month"] = data["Order_Date"].dt.month
    data["Order_Day"] = data["Order_Date"].dt.day
    data["Order_Year"] = data["Order_Date"].dt.year
    return data

def train_and_save():
    df = make_features(load_data())
    # Avoid post-transaction / target-derived columns to reduce data leakage.
    X = df[FEATURES]
    y = df[TARGET]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL)
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Compare several regression algorithms on the same held-out test set.
    candidates = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=80, random_state=42, min_samples_leaf=2, n_jobs=1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=80, random_state=42, min_samples_leaf=2, n_jobs=1
        )
    }
    results, fitted = [], {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results.append({
            "model": name,
            "MAE": float(mean_absolute_error(y_test, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "R2": float(r2_score(y_test, pred))
        })
        fitted[name] = pipe

    # Select the model with the lowest MAE, then refit it on the training split.
    metrics = pd.DataFrame(results).sort_values("MAE")
    best_name = metrics.iloc[0]["model"]
    best_model = fitted[best_name]
    joblib.dump(best_model, MODEL_DIR / "sales_model.joblib")
    metrics.to_csv(MODEL_DIR / "model_metrics.csv", index=False)
    with open(MODEL_DIR / "model_info.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_model": best_name,
            "target": TARGET,
            "features": FEATURES,
            "test_size": 0.20,
            "random_state": 42,
            "train_rows": len(X_train),
            "test_rows": len(X_test)
        }, f, indent=2)
    return best_model, metrics, best_name

if __name__ == "__main__":
    model, metrics, best = train_and_save()
    print("\\nModel comparison:")
    print(metrics.to_string(index=False))
    print(f"\\nSelected model: {best}")
    print(f"Saved model to: {MODEL_DIR / 'sales_model.joblib'}")
