# ============================================
# FINAL ROBUST CARBON EMISSION MODEL
# - Combines BOTH datasets
# - Uses Ensemble Learning
# - Performs Hyperparameter Tuning
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
import joblib

# =========================
# 1️⃣ LOAD BOTH DATASETS
# =========================
df1 = pd.read_csv("datasetkaggle1.csv")
df2 = pd.read_csv("new_test_dataset.csv")

# Combine datasets
df = pd.concat([df1, df2], ignore_index=True)
df = df.dropna()

target_column = "CarbonEmission"

# Encode categorical features
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop(target_column, axis=1)
y = df_encoded[target_column]

# =========================
# 2️⃣ TRAIN TEST SPLIT (70/30)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# =========================
# 3️⃣ DEFINE BASE MODELS
# =========================

rf = RandomForestRegressor(random_state=42)
gb = GradientBoostingRegressor(random_state=42)
en = ElasticNet(random_state=42)

# =========================
# 4️⃣ CREATE ENSEMBLE
# =========================

ensemble = VotingRegressor(
    estimators=[
        ("rf", rf),
        ("gb", gb),
        ("en", en)
    ]
)

# =========================
# 5️⃣ HYPERPARAMETER TUNING
# =========================

param_dist = {
    "rf__n_estimators": [100, 200, 300],
    "rf__max_depth": [None, 10, 20],
    "gb__n_estimators": [100, 200],
    "gb__learning_rate": [0.05, 0.1],
    "en__alpha": [0.01, 0.1, 1.0],
}

random_search = RandomizedSearchCV(
    ensemble,
    param_distributions=param_dist,
    n_iter=15,
    cv=3,
    scoring="r2",
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

print("Best Parameters Found:")
print(random_search.best_params_)

# =========================
# 6️⃣ FINAL EVALUATION
# =========================

predictions = best_model.predict(X_test)

print("\n===== FINAL MODEL PERFORMANCE =====")
print("R2 Score:", r2_score(y_test, predictions))
print("Mean Squared Error:", mean_squared_error(y_test, predictions))

# =========================
# 7️⃣ SAVE FINAL MODEL
# =========================

joblib.dump(best_model, "final_carbon_model.pkl")
print("\nFinal model saved successfully!")
import matplotlib.pyplot as plt

residuals = y_test - predictions

plt.hist(residuals, bins=50)
plt.title("Residual Distribution")
plt.show()
import pandas as pd

importance = best_model.named_estimators_['rf'].feature_importances_
features = X.columns

feature_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print(feature_df.head(10))
joblib.dump(best_model, "final_carbon_model.pkl")
