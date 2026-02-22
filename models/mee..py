# import pandas as pd
# import numpy as np

# np.random.seed(42)
# n = 3000

# # Brand emission factors (kg CO2 per month of usage)
# brand_emission_factors = {
#     'Amazon':        8.5,
#     'Zomato/Swiggy': 4.2,
#     'Uber/Ola':      6.8,
#     'McDonalds/KFC': 9.1,
#     'H&M/Zara':      11.3,
#     'Apple/Samsung': 15.0,
#     'Netflix':       1.2,
#     'None':          0.0
# }

# brands = list(brand_emission_factors.keys())

# # Each user can frequently use 1 to 3 brands
# def assign_brand_emission(n):
#     total_brand_co2 = []
#     selected_brands = []
#     for _ in range(n):
#         num_brands = np.random.randint(1, 4)  # 1 to 3 brands per user
#         chosen = np.random.choice(brands, num_brands, replace=False)
#         co2 = sum(brand_emission_factors[b] for b in chosen)
#         total_brand_co2.append(co2)
#         selected_brands.append(', '.join(chosen))
#     return total_brand_co2, selected_brands

# brand_co2, brand_labels = assign_brand_emission(n)

# df = pd.DataFrame({
#     'electricity_kwh':  np.random.uniform(50, 1000, n),
#     'lpg_kg':           np.random.uniform(0, 20, n),
#     'fuel_litres':      np.random.uniform(0, 200, n),
#     'flights_year':     np.random.randint(0, 20, n),
#     'public_transport': np.random.randint(0, 60, n),
#     'meat_days_week':   np.random.randint(0, 7, n),
#     'waste_kg_week':    np.random.uniform(1, 20, n),
#     'brands_used':      brand_labels,
#     'brand_co2':        brand_co2
# })

# # Calculate target variable
# df['total_CO2'] = (
#     df['electricity_kwh']  * 0.82 +
#     df['lpg_kg']           * 2.98 +
#     df['fuel_litres']      * 2.3  +
#     (df['flights_year'] / 12) * 90 +
#     df['meat_days_week']   * 7    +
#     df['waste_kg_week']    * 1.5  * 4 +
#     df['brand_co2']                    # brand contribution
# )

# # Add realistic noise
# df['total_CO2'] += np.random.normal(0, 20, n)
# df['total_CO2'] = df['total_CO2'].clip(lower=0)

# # Classification label
# df['category'] = pd.cut(
#     df['total_CO2'],
#     bins=[0, 300, 700, float('inf')],
#     labels=['Low', 'Medium', 'High']
# )

# df.to_csv('carbon_footprint_dataset.csv', index=False)
# print(df.describe())
# print(df['category'].value_counts())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)

# ─────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────
df = pd.read_csv('carbon_footprint_dataset.csv')

# Drop the string brand column — model only needs numeric brand_co2
df = df.drop(columns=['brands_used'])

# Encode category label: Low=0, Medium=1, High=2
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

print("Dataset shape:", df.shape)
print(df['category'].value_counts())

# ─────────────────────────────────────────
# 2. DEFINE FEATURES & TARGETS
# ─────────────────────────────────────────
features = [
    'electricity_kwh',
    'lpg_kg',
    'fuel_litres',
    'flights_year',
    'public_transport',
    'meat_days_week',
    'waste_kg_week',
    'brand_co2'
]

X = df[features]

y_reg   = df['total_CO2']           # Regression target
y_clf   = df['category_encoded']    # Classification target

# ─────────────────────────────────────────
# 3. TRAIN-TEST SPLIT (80/20)
# ─────────────────────────────────────────
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf,
    test_size=0.2,
    random_state=42
)

print(f"\nTraining samples: {len(X_train)} | Testing samples: {len(X_test)}")

# ─────────────────────────────────────────
# 4. LINEAR REGRESSION (CO2 Prediction)
# ─────────────────────────────────────────
reg_model = LinearRegression()
reg_model.fit(X_train, y_reg_train)

y_reg_pred = reg_model.predict(X_test)

r2  = r2_score(y_reg_test, y_reg_pred)
mae = mean_absolute_error(y_reg_test, y_reg_pred)

print("\n── Linear Regression Results ──")
print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.2f} kg CO2")

# Coefficients (feature importance proxy for linear regression)
coef_df = pd.DataFrame({
    'Feature':     features,
    'Coefficient': reg_model.coef_
}).sort_values('Coefficient', ascending=False)

print("\nFeature Coefficients:")
print(coef_df)

# ─────────────────────────────────────────
# 5. DECISION TREE CLASSIFIER (Category)
# ─────────────────────────────────────────
clf_model = DecisionTreeClassifier(
    max_depth=6,          # prevents overfitting
    min_samples_split=10,
    random_state=42
)
clf_model.fit(X_train, y_clf_train)

y_clf_pred = clf_model.predict(X_test)

print("\n── Decision Tree Classifier Results ──")
print(classification_report(
    y_clf_test, y_clf_pred,
    target_names=le.classes_
))

# ─────────────────────────────────────────
# 6. VISUALIZATIONS
# ─────────────────────────────────────────

# Plot 1: Actual vs Predicted CO2 (Regression)
plt.figure(figsize=(7, 5))
plt.scatter(y_reg_test, y_reg_pred, alpha=0.4, color='steelblue', edgecolors='k', linewidths=0.3)
plt.plot([y_reg_test.min(), y_reg_test.max()],
         [y_reg_test.min(), y_reg_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual CO2 (kg/month)')
plt.ylabel('Predicted CO2 (kg/month)')
plt.title('Linear Regression: Actual vs Predicted CO2')
plt.legend()
plt.tight_layout()
plt.savefig('regression_actual_vs_predicted.png')
plt.show()

# Plot 2: Feature Coefficients
plt.figure(figsize=(8, 5))
sns.barplot(data=coef_df, x='Coefficient', y='Feature', palette='coolwarm')
plt.title('Feature Coefficients (Linear Regression)')
plt.tight_layout()
plt.savefig('feature_coefficients.png')
plt.show()

# Plot 3: Confusion Matrix (Classifier)
cm = confusion_matrix(y_clf_test, y_clf_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Decision Tree: Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Plot 4: Decision Tree Structure
plt.figure(figsize=(20, 8))
plot_tree(clf_model, feature_names=features,
          class_names=le.classes_,
          filled=True, rounded=True, fontsize=9)
plt.title('Decision Tree Structure')
plt.tight_layout()
plt.savefig('decision_tree_structure.png')
plt.show()

# ─────────────────────────────────────────
# 7. SAVE MODELS
# ─────────────────────────────────────────
joblib.dump(reg_model, 'linear_regression_co2.pkl')
joblib.dump(clf_model, 'decision_tree_category.pkl')
joblib.dump(le,        'label_encoder.pkl')

print("\n✅ Models saved successfully!")