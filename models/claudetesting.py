import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ─────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────
DATA_PATH   = "new_test_dataset.csv"        # <-- update path if needed
MODEL_PATH  = "trained_model.pkl"           # <-- your saved model file (if any)
TARGET_COL  = "CarbonEmission"

df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(df.head(3))

# ─────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────
# Drop columns that are purely list-like strings (Recycling, Cooking_With)
# or encode them as simple string features
def clean_list_col(series):
    """Convert list-like strings to a simple comma-separated string."""
    return series.astype(str).str.replace(r"[\[\]']", "", regex=True).str.strip()

list_cols = ["Recycling", "Cooking_With"]
for col in list_cols:
    if col in df.columns:
        df[col] = clean_list_col(df[col])

# Encode all categorical (object) columns with LabelEncoder
le_map = {}
for col in df.select_dtypes(include="object").columns:
    if col == TARGET_COL:
        continue
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_map[col] = le

# Fill any remaining NaNs
df.fillna(df.median(numeric_only=True), inplace=True)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

print(f"\nFeatures: {list(X.columns)}")
print(f"Target  : {TARGET_COL}  |  range: [{y.min()}, {y.max()}]")

# ─────────────────────────────────────────
# 3. LOAD OR TRAIN MODEL
# ─────────────────────────────────────────
if os.path.exists(MODEL_PATH):
    print(f"\n✅ Loading existing model from: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Use the full dataset as the test set (since the model was already trained)
    X_test, y_test = X, y
    y_pred = model.predict(X_test)

else:
    print(f"\n⚠️  No saved model found at '{MODEL_PATH}'.")
    print("   Training a new Gradient Boosting Regressor on 80/20 split...\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Save the model for future use
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to: {MODEL_PATH}")

# ─────────────────────────────────────────
# 4. ACCURACY / EVALUATION METRICS
# ─────────────────────────────────────────
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

# Mean Absolute Percentage Error
mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test == 0, 1, y_test))) * 100

print("\n" + "="*45)
print("       MODEL EVALUATION RESULTS")
print("="*45)
print(f"  Samples evaluated : {len(y_test)}")
print(f"  R² Score          : {r2:.4f}   (1.0 = perfect)")
print(f"  MAE               : {mae:.2f}")
print(f"  RMSE              : {rmse:.2f}")
print(f"  MAPE              : {mape:.2f}%")
print("="*45)

# ─────────────────────────────────────────
# 5. SAMPLE PREDICTIONS
# ─────────────────────────────────────────
results_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred.round(0),
    "Error": (y_test.values - y_pred).round(2)
})
print("\nSample Predictions (first 10 rows):")
print(results_df.head(10).to_string(index=False))

# ─────────────────────────────────────────
# 6. FEATURE IMPORTANCE (if supported)
# ─────────────────────────────────────────
if hasattr(model, "feature_importances_"):
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    print("\nTop 10 Feature Importances:")
    print(importance_df.head(10).to_string(index=False))
    