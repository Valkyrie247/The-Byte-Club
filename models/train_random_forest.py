# ============================================
# Carbon Emission Prediction using Random Forest
# Train: 70% | Test: 30%
# ============================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load Dataset
# Make sure datasetkaggle1.csv is in the same folder as this script
df = pd.read_csv("datasetkaggle1.csv")

# Step 2: Remove missing values (simple clean method)
df = df.dropna()

# Step 3: Define Target Column
target_column = "CarbonEmission"

# Step 4: Convert categorical columns to numeric
df_encoded = pd.get_dummies(df, drop_first=True)

# Step 5: Split Features and Target
X = df_encoded.drop(target_column, axis=1)
y = df_encoded[target_column]

# Step 6: Train-Test Split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Step 7: Train Random Forest Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Step 8: Make Predictions
predictions = model.predict(X_test)

# Step 9: Evaluate Model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("===== Random Forest Model Results =====")
print("Training Size:", len(X_train))
print("Testing Size:", len(X_test))
print("Mean Squared Error (MSE):", mse)
print("R2 Score (Accuracy Measure):", r2)

# Optional: Show sample predictions
print("\nSample Predictions vs Actual:")
for i in range(5):
    print(f"Actual: {y_test.iloc[i]}  |  Predicted: {round(predictions[i],2)}")