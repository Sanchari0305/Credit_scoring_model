# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             classification_report, roc_curve)
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# 2. Load Data
csv_path = 'C:/Users/T U F/Documents/dataset/archive (1)/german_credit_cleaned.csv'
df = pd.read_csv(csv_path)
print(f"Data loaded: {df.shape}")

# 3. Map 'saving_acc_bonds' to Numeric (update mapping as needed)
saving_acc_map = {
    'unknown_no_saving_acc': 0,
    'below_100': 1,
    'below_500': 2,
    'below_1000': 3,
    'above_1000': 4,
}
df['saving_acc_bonds_mapped'] = df['saving_acc_bonds'].map(saving_acc_map)
print("Nulls after mapping:", df['saving_acc_bonds_mapped'].isnull().sum())

# 4. Ensure Numeric Columns for Feature Engineering
for col in ['num_curr_loans', 'num_people_provide_maint', 'loan_amt', 'age']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 5. Feature Engineering
df['dti'] = df['loan_amt'] / (df['saving_acc_bonds_mapped'].replace(0, 1) + 1e-6)
df['credit_util'] = df['num_curr_loans'] / (df['num_people_provide_maint'] + 1)
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 100], labels=['<30', '30-40', '40-50', '50+'])

# 6. Encode Categorical Variables
le = LabelEncoder()
for col in df.select_dtypes(include=['object', 'category']).columns:
    if col != 'target':
        df[col] = le.fit_transform(df[col].astype(str))

# 7. Encode Target Variable
df['target'] = df['target'].map({'good': 1, 'bad': 0})

# 8. Drop rows with missing values (if any remain after conversion)
df = df.dropna()
print("Final shape after dropna:", df.shape)

# 9. Split Data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 10. Balance Classes with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 11. Feature Scaling
scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# 12. Train Models with STRONGER Class Weights
lr = LogisticRegression(max_iter=2000, class_weight={0: 3, 1: 1}, random_state=42)
lr.fit(X_train_res_scaled, y_train_res)

dt = DecisionTreeClassifier(class_weight={0: 3, 1: 1}, random_state=42)
dt.fit(X_train_res, y_train_res)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_leaf': [1, 2, 4]
}
rf_base = RandomForestClassifier(class_weight={0: 5, 1: 1}, random_state=42)
grid = GridSearchCV(rf_base, param_grid, scoring='f1', cv=5, n_jobs=-1)
grid.fit(X_train_res, y_train_res)
rf = grid.best_estimator_

print("Best Random Forest parameters:", grid.best_params_)

# 13. Evaluate Models (Standard Threshold 0.5)
models = {
    'Logistic Regression': (lr, X_test_scaled),
    'Decision Tree': (dt, X_test),
    'Random Forest': (rf, X_test)
}

for name, (model, X_eval) in models.items():
    y_pred = model.predict(X_eval)
    print(f"\n{name} (Standard Threshold 0.5):")
    print(classification_report(y_test, y_pred))

# 14. Business-Optimized Threshold Tuning
rf_proba_bad = rf.predict_proba(X_test)[:, 0]

def calculate_profit(y_true, y_pred):
    tp_bad = np.sum((y_true == 0) & (y_pred == 0))
    fp_good = np.sum((y_true == 1) & (y_pred == 0))
    fn_bad = np.sum((y_true == 0) & (y_pred == 1))
    tn_good = np.sum((y_true == 1) & (y_pred == 1))
    
    # UPDATED BUSINESS PARAMETERS
    loss_per_bad = 8000    # Loss from bad loan
    gain_per_good = 3000   # Profit from good customer (INCREASED)
    cost_reject_good = 1000 # Opportunity cost (INCREASED)
    
    profit = (tp_bad * loss_per_bad) + (tn_good * gain_per_good) - (fp_good * cost_reject_good) - (fn_bad * loss_per_bad)
    return profit
# Test thresholds
thresholds = np.linspace(0.1, 0.9, 100)
profits = []

for t in thresholds:
    y_pred_temp = (rf_proba_bad > t).astype(int)
    profits.append(calculate_profit(y_test, y_pred_temp))

best_threshold = thresholds[np.argmax(profits)]
max_profit = max(profits)

# Apply best threshold
y_pred_custom = (rf_proba_bad > best_threshold).astype(int)

print(f"\nRandom Forest (Business-Optimized Threshold={best_threshold:.3f}):")
print(classification_report(y_test, y_pred_custom))
print(f"Estimated Profit: ${max_profit}")

# 15. Profit Curve Visualization
plt.figure(figsize=(10,6))
plt.plot(thresholds, profits, 'b-', label='Profit')
plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Optimal Threshold: {best_threshold:.3f}')
plt.xlabel('Threshold')
plt.ylabel('Profit ($)')
plt.title('Profit Optimization Curve')
plt.legend()
plt.grid(True)
plt.show()

# 16. Feature Importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12,6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()
 