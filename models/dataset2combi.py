import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    classification_report, confusion_matrix,
    accuracy_score
)

# ═══════════════════════════════════════════════════════
# SECTION 1: BRAND EMISSION SETUP
# ═══════════════════════════════════════════════════════
np.random.seed(42)
n = 3000

brand_emission_factors = {
    'Amazon':        8.5,
    'Zomato/Swiggy': 4.2,
    'Uber/Ola':      6.8,
    'McDonalds/KFC': 9.1,
    'H&M/Zara':      11.3,
    'Apple/Samsung': 15.0,
    'Netflix':       1.2,
    'None':          0.0
}
brands = list(brand_emission_factors.keys())

def assign_brand_emission(n):
    total_brand_co2, selected_brands = [], []
    for _ in range(n):
        num_brands = np.random.randint(1, 4)
        chosen = np.random.choice(brands, num_brands, replace=False)
        co2 = sum(brand_emission_factors[b] for b in chosen)
        total_brand_co2.append(co2)
        selected_brands.append(', '.join(chosen))
    return total_brand_co2, selected_brands

# ═══════════════════════════════════════════════════════
# SECTION 2: DATASET GENERATION
# ═══════════════════════════════════════════════════════

# ── Dataset 1: Personal Lifestyle ──
print("⏳ Generating Dataset 1: Personal Lifestyle...")
brand_co2, _ = assign_brand_emission(n)

df1 = pd.DataFrame({
    'electricity_kwh':  np.random.uniform(50, 1000, n),
    'lpg_kg':           np.random.uniform(0, 20, n),
    'fuel_litres':      np.random.uniform(0, 200, n),
    'flights_year':     np.random.randint(0, 20, n),
    'public_transport': np.random.randint(0, 60, n),
    'meat_days_week':   np.random.randint(0, 7, n),
    'waste_kg_week':    np.random.uniform(1, 20, n),
    'brand_co2':        brand_co2,
    'vehicle_type_co2': np.zeros(n),
    'commute_km':       np.random.uniform(0, 20, n),
    'diet_type_co2':    np.zeros(n),
    'food_waste_kg':    np.random.uniform(0, 10, n),
    'industry_kwh':     np.zeros(n),
    'industrial_waste': np.zeros(n),
    'country_co2_pc':   np.random.uniform(1, 10, n),
    'source':           'personal_lifestyle'
})
df1['total_CO2'] = (
    df1['electricity_kwh']  * 0.82 +
    df1['lpg_kg']           * 2.98 +
    df1['fuel_litres']      * 2.3  +
    (df1['flights_year'] / 12) * 90 +
    df1['meat_days_week']   * 7    +
    df1['waste_kg_week']    * 1.5  * 4 +
    df1['brand_co2']        +
    df1['commute_km']       * 0.21 * 22
)
df1['total_CO2'] += np.random.normal(0, 20, n)
df1['total_CO2']  = df1['total_CO2'].clip(lower=0)
print(f"✅ Dataset 1 created: {df1.shape}")

# ── Dataset 2: Transport & Commute ──
print("\n⏳ Generating Dataset 2: Transport & Commute...")
vehicle_types = {
    'Petrol Car':   0.21,
    'Diesel Car':   0.27,
    'Electric Car': 0.05,
    'Motorbike':    0.11,
    'Bus':          0.04,
    'None':         0.0
}
vehicle_choices  = np.random.choice(list(vehicle_types.keys()), n)
vehicle_type_co2 = np.array([vehicle_types[v] for v in vehicle_choices])
commute_km       = np.random.uniform(5, 100, n)
brand_co2, _     = assign_brand_emission(n)

df2 = pd.DataFrame({
    'electricity_kwh':  np.random.uniform(50, 400, n),
    'lpg_kg':           np.random.uniform(0, 10, n),
    'fuel_litres':      commute_km * 0.08 * 22,
    'flights_year':     np.random.randint(0, 30, n),
    'public_transport': np.random.randint(0, 90, n),
    'meat_days_week':   np.random.randint(0, 7, n),
    'waste_kg_week':    np.random.uniform(1, 10, n),
    'brand_co2':        brand_co2,
    'vehicle_type_co2': vehicle_type_co2,
    'commute_km':       commute_km,
    'diet_type_co2':    np.zeros(n),
    'food_waste_kg':    np.random.uniform(0, 5, n),
    'industry_kwh':     np.zeros(n),
    'industrial_waste': np.zeros(n),
    'country_co2_pc':   np.random.uniform(1, 10, n),
    'source':           'transport_commute'
})
df2['total_CO2'] = (
    df2['electricity_kwh']   * 0.82 +
    df2['lpg_kg']            * 2.98 +
    df2['fuel_litres']       * 2.3  +
    (df2['flights_year'] / 12) * 90 +
    df2['commute_km'] * df2['vehicle_type_co2'] * 22 +
    df2['meat_days_week']    * 7    +
    df2['waste_kg_week']     * 1.5  * 4 +
    df2['brand_co2']
)
df2['total_CO2'] += np.random.normal(0, 25, n)
df2['total_CO2']  = df2['total_CO2'].clip(lower=0)
print(f"✅ Dataset 2 created: {df2.shape}")

# ── Dataset 3: Food & Diet ──
print("\n⏳ Generating Dataset 3: Food & Diet...")
diet_types = {
    'Vegan':        1.5,
    'Vegetarian':   2.5,
    'Pescatarian':  3.5,
    'Omnivore':     6.0,
    'Heavy Meat':   9.5
}
diet_choices = np.random.choice(list(diet_types.keys()), n)
diet_co2     = np.array([diet_types[d] for d in diet_choices])
brand_co2, _ = assign_brand_emission(n)

df3 = pd.DataFrame({
    'electricity_kwh':  np.random.uniform(50, 600, n),
    'lpg_kg':           np.random.uniform(0, 15, n),
    'fuel_litres':      np.random.uniform(0, 100, n),
    'flights_year':     np.random.randint(0, 10, n),
    'public_transport': np.random.randint(0, 60, n),
    'meat_days_week':   np.random.randint(0, 7, n),
    'waste_kg_week':    np.random.uniform(1, 15, n),
    'brand_co2':        brand_co2,
    'vehicle_type_co2': np.zeros(n),
    'commute_km':       np.random.uniform(0, 30, n),
    'diet_type_co2':    diet_co2,
    'food_waste_kg':    np.random.uniform(0, 15, n),
    'industry_kwh':     np.zeros(n),
    'industrial_waste': np.zeros(n),
    'country_co2_pc':   np.random.uniform(1, 10, n),
    'source':           'food_diet'
})
df3['total_CO2'] = (
    df3['electricity_kwh']  * 0.82 +
    df3['lpg_kg']           * 2.98 +
    df3['fuel_litres']      * 2.3  +
    (df3['flights_year'] / 12) * 90 +
    df3['diet_type_co2']    * 30   +
    df3['food_waste_kg']    * 2.5  * 4 +
    df3['waste_kg_week']    * 1.5  * 4 +
    df3['brand_co2']
)
df3['total_CO2'] += np.random.normal(0, 20, n)
df3['total_CO2']  = df3['total_CO2'].clip(lower=0)
print(f"✅ Dataset 3 created: {df3.shape}")

# ── Dataset 4: Industry / Corporate ──
print("\n⏳ Generating Dataset 4: Industry / Corporate...")
industry_sectors = {
    'Manufacturing': 2.5,
    'IT/Software':   0.8,
    'Retail':        1.2,
    'Healthcare':    1.5,
    'Construction':  3.0,
    'Agriculture':   2.0
}
sector_choices   = np.random.choice(list(industry_sectors.keys()), n)
sector_factor    = np.array([industry_sectors[s] for s in sector_choices])
industry_kwh     = np.random.uniform(500, 10000, n)
industrial_waste = np.random.uniform(10, 500, n)
brand_co2, _     = assign_brand_emission(n)

df4 = pd.DataFrame({
    'electricity_kwh':  np.random.uniform(100, 800, n),
    'lpg_kg':           np.random.uniform(0, 30, n),
    'fuel_litres':      np.random.uniform(50, 300, n),
    'flights_year':     np.random.randint(0, 50, n),
    'public_transport': np.random.randint(0, 30, n),
    'meat_days_week':   np.random.randint(0, 7, n),
    'waste_kg_week':    np.random.uniform(5, 50, n),
    'brand_co2':        brand_co2,
    'vehicle_type_co2': np.zeros(n),
    'commute_km':       np.random.uniform(10, 80, n),
    'diet_type_co2':    np.zeros(n),
    'food_waste_kg':    np.random.uniform(0, 20, n),
    'industry_kwh':     industry_kwh,
    'industrial_waste': industrial_waste,
    'country_co2_pc':   np.random.uniform(1, 15, n),
    'source':           'industry_corporate'
})
df4['total_CO2'] = (
    df4['electricity_kwh']  * 0.82 +
    df4['lpg_kg']           * 2.98 +
    df4['fuel_litres']      * 2.3  +
    (df4['flights_year'] / 12) * 90 +
    df4['industry_kwh']     * sector_factor / 100 +
    df4['industrial_waste'] * 1.8  +
    df4['waste_kg_week']    * 1.5  * 4 +
    df4['brand_co2']
)
df4['total_CO2'] += np.random.normal(0, 50, n)
df4['total_CO2']  = df4['total_CO2'].clip(lower=0)
print(f"✅ Dataset 4 created: {df4.shape}")

# ═══════════════════════════════════════════════════════
# SECTION 3: MERGE ALL DATASETS
# ═══════════════════════════════════════════════════════
print("\n⏳ Merging all datasets...")

df_all = pd.concat([df1, df2, df3, df4], ignore_index=True)
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

# Check CO2 distribution before binning
print(f"\nCO2 Stats:\n  Min: {df_all['total_CO2'].min():.1f}")
print(f"  Max: {df_all['total_CO2'].max():.1f}")
print(f"  Mean: {df_all['total_CO2'].mean():.1f}")
print(f"  Median: {df_all['total_CO2'].median():.1f}")

# Use percentile-based bins for balanced classes
low_thresh  = df_all['total_CO2'].quantile(0.33)
high_thresh = df_all['total_CO2'].quantile(0.66)
print(f"\nAuto bins → Low < {low_thresh:.1f} | Medium < {high_thresh:.1f} | High >= {high_thresh:.1f}")

df_all['category'] = pd.cut(
    df_all['total_CO2'],
    bins=[0, low_thresh, high_thresh, float('inf')],
    labels=['Low', 'Medium', 'High']
)
df_all = df_all.dropna(subset=['category'])
df_all.to_csv('merged_carbon_dataset.csv', index=False)

print(f"\n✅ Merged Dataset Shape : {df_all.shape}")
print("\nCategory Distribution:")
print(df_all['category'].value_counts())
print("\nSource Distribution:")
print(df_all['source'].value_counts())

# ═══════════════════════════════════════════════════════
# SECTION 4: PREPROCESSING
# ═══════════════════════════════════════════════════════
print("\n⏳ Preprocessing...")

le = LabelEncoder()
df_all['category_encoded'] = le.fit_transform(df_all['category'])

features = [
    'electricity_kwh',
    'lpg_kg',
    'fuel_litres',
    'flights_year',
    'public_transport',
    'meat_days_week',
    'waste_kg_week',
    'brand_co2',
    'vehicle_type_co2',
    'commute_km',
    'diet_type_co2',
    'food_waste_kg',
    'industry_kwh',
    'industrial_waste',
    'country_co2_pc'
]

X     = df_all[features]
y_reg = df_all['total_CO2']
y_clf = df_all['category_encoded']

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf,
    test_size=0.2,
    random_state=42,
    stratify=y_clf
)

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# ═══════════════════════════════════════════════════════
# SECTION 5: LINEAR REGRESSION
# ═══════════════════════════════════════════════════════
print("\n⏳ Training Linear Regression...")

reg_model  = LinearRegression()
reg_model.fit(X_train, y_reg_train)
y_reg_pred = reg_model.predict(X_test)

r2  = r2_score(y_reg_test, y_reg_pred)
mae = mean_absolute_error(y_reg_test, y_reg_pred)

# Regression Accuracy — within ±10% tolerance
tolerance        = 0.10
within_tolerance = np.abs(y_reg_pred - y_reg_test) <= (tolerance * np.abs(y_reg_test))
reg_accuracy     = within_tolerance.mean() * 100

print("\n╔══════════════════════════════════════════╗")
print("║       LINEAR REGRESSION RESULTS          ║")
print("╠══════════════════════════════════════════╣")
print(f"║  R² Score          : {r2:.4f}              ║")
print(f"║  MAE               : {mae:.2f} kg CO2/month ║")
print(f"║  Accuracy (±10%)   : {reg_accuracy:.2f}%             ║")
print("╚══════════════════════════════════════════╝")

coef_df = pd.DataFrame({
    'Feature':     features,
    'Coefficient': reg_model.coef_
}).sort_values('Coefficient', ascending=False)

print("\nFeature Coefficients (sorted):")
print(coef_df.to_string(index=False))

# ═══════════════════════════════════════════════════════
# SECTION 6: DECISION TREE CLASSIFIER
# ═══════════════════════════════════════════════════════
print("\n⏳ Training Decision Tree Classifier...")

clf_model  = DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=15,
    class_weight='balanced',
    random_state=42
)
clf_model.fit(X_train, y_clf_train)
y_clf_pred = clf_model.predict(X_test)

clf_accuracy = accuracy_score(y_clf_test, y_clf_pred) * 100

print("\n╔══════════════════════════════════════════╗")
print("║    DECISION TREE CLASSIFIER RESULTS      ║")
print("╠══════════════════════════════════════════╣")
print(f"║  Accuracy          : {clf_accuracy:.2f}%             ║")
print("╚══════════════════════════════════════════╝")

print("\nDetailed Classification Report:")
print(classification_report(
    y_clf_test, y_clf_pred,
    target_names=le.classes_
))

# ═══════════════════════════════════════════════════════
# SECTION 7: VISUALIZATIONS
# ═══════════════════════════════════════════════════════
print("\n⏳ Generating visualizations...")

# 1. Actual vs Predicted CO2
plt.figure(figsize=(7, 5))
plt.scatter(y_reg_test, y_reg_pred, alpha=0.3,
            color='steelblue', edgecolors='k', linewidths=0.2)
plt.plot([y_reg_test.min(), y_reg_test.max()],
         [y_reg_test.min(), y_reg_test.max()],
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual CO2 (kg/month)')
plt.ylabel('Predicted CO2 (kg/month)')
plt.title(f'Linear Regression: Actual vs Predicted\nR²={r2:.4f} | MAE={mae:.2f} | Accuracy={reg_accuracy:.1f}%')
plt.legend()
plt.tight_layout()
plt.savefig('regression_actual_vs_predicted.png', dpi=150)
plt.close()
print("✅ Plot 1 saved")

# 2. Feature Coefficients
plt.figure(figsize=(9, 6))
sns.barplot(data=coef_df, x='Coefficient', y='Feature', palette='coolwarm')
plt.title('Feature Coefficients — Linear Regression')
plt.tight_layout()
plt.savefig('feature_coefficients.png', dpi=150)
plt.close()
print("✅ Plot 2 saved")

# 3. Confusion Matrix
cm = confusion_matrix(y_clf_test, y_clf_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Decision Tree Confusion Matrix\nAccuracy: {clf_accuracy:.2f}%')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()
print("✅ Plot 3 saved")

# 4. CO2 Distribution by Source
plt.figure(figsize=(9, 5))
sns.boxplot(data=df_all, x='source', y='total_CO2', palette='Set2')
plt.xticks(rotation=15)
plt.title('CO2 Distribution by Dataset Source')
plt.tight_layout()
plt.savefig('co2_by_source.png', dpi=150)
plt.close()
print("✅ Plot 4 saved")

# 5. Category Distribution
plt.figure(figsize=(6, 4))
counts = df_all['category'].value_counts().sort_index()
bars   = plt.bar(counts.index, counts.values, color=['green', 'orange', 'red'])
plt.title('Emission Category Distribution')
plt.xlabel('Category')
plt.ylabel('Count')
for bar, val in zip(bars, counts.values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 20,
             str(val), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('category_distribution.png', dpi=150)
plt.close()
print("✅ Plot 5 saved")

# 6. Accuracy Summary
plt.figure(figsize=(6, 4))
metrics = ['Regression\nAccuracy (±10%)', 'Classifier\nAccuracy']
values  = [reg_accuracy, clf_accuracy]
colors  = ['steelblue', 'seagreen']
bars_ax = plt.bar(metrics, values, color=colors, width=0.4)
for bar, val in zip(bars_ax, values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() - 4,
             f'{val:.2f}%', ha='center', va='top',
             color='white', fontweight='bold', fontsize=12)
plt.ylim(0, 110)
plt.title('Model Accuracy Summary')
plt.ylabel('Accuracy (%)')
plt.tight_layout()
plt.savefig('accuracy_summary.png', dpi=150)
plt.close()
print("✅ Plot 6 saved")

# ═══════════════════════════════════════════════════════
# SECTION 8: SAVE MODELS & DATASET
# ═══════════════════════════════════════════════════════
joblib.dump(reg_model, 'linear_regression_co2.pkl')
joblib.dump(clf_model, 'decision_tree_category.pkl')
joblib.dump(le,        'label_encoder.pkl')

print("\n✅ All models and dataset saved!")
print("\n═══════════════ FINAL SUMMARY ═══════════════")
print(f"  Total Records          : {len(df_all)}")
print(f"  Training Samples       : {len(X_train)}")
print(f"  Testing Samples        : {len(X_test)}")
print(f"  Regression R²          : {r2:.4f}")
print(f"  Regression MAE         : {mae:.2f} kg CO2/month")
print(f"  Regression Accuracy±10%: {reg_accuracy:.2f}%")
print(f"  Classifier Accuracy    : {clf_accuracy:.2f}%")
print("═════════════════════════════════════════════")