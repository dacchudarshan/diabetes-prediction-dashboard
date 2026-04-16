import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. Load dataset
df = pd.read_csv('diabetes_dataset.csv')
print("Dataset:\n", df.head())
print(f"Dataset shape: {df.shape}\n")

# Sample data for faster training (use 70,000 samples)
df = df.sample(n=min(70000, len(df)), random_state=42)
print(f"Sampled dataset shape: {df.shape}\n")

# 2. Features & Target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Encode categorical columns
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])

# Scale features for better performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data for regression
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# For classification (Logistic Regression), use binary target
y_binary = (y > y.mean()).astype(int)
_, _, y_train_binary, y_test_binary = train_test_split(X_scaled, y_binary, test_size=0.2, random_state=42)

# Store results for comparison
results = {}

# ===== MODEL 1: LOGISTIC REGRESSION =====
print("\n" + "="*50)
print("1️⃣  LOGISTIC REGRESSION")
print("="*50)
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train_binary)
y_pred_log = log_model.predict(X_test)

acc_log = accuracy_score(y_test_binary, y_pred_log)
mse_log = mean_squared_error(y_test_binary, y_pred_log)
mae_log = mean_absolute_error(y_test_binary, y_pred_log)
cv_log = cross_val_score(log_model, X_train, y_train_binary, cv=5, scoring='accuracy').mean()

print(f"Accuracy: {acc_log:.4f}")
print(f"Mean Squared Error: {mse_log:.4f}")
print(f"Mean Absolute Error: {mae_log:.4f}")
print(f"Cross-Validation Score: {cv_log:.4f}")
results['Logistic Regression'] = {'accuracy': acc_log, 'mse': mse_log, 'mae': mae_log, 'r2': acc_log}

# ===== MODEL 2: MULTIPLE LINEAR REGRESSION =====
print("\n" + "="*50)
print("2️⃣  MULTIPLE LINEAR REGRESSION")
print("="*50)
mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)
y_pred_mlr = mlr_model.predict(X_test)

mse_mlr = mean_squared_error(y_test, y_pred_mlr)
mae_mlr = mean_absolute_error(y_test, y_pred_mlr)
r2_mlr = r2_score(y_test, y_pred_mlr)
rmse_mlr = np.sqrt(mse_mlr)

print(f"Mean Squared Error (MSE): {mse_mlr:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_mlr:.4f}")
print(f"Mean Absolute Error (MAE): {mae_mlr:.4f}")
print(f"R² Score: {r2_mlr:.4f}")
results['Multiple Linear Regression'] = {'accuracy': r2_mlr, 'mse': mse_mlr, 'mae': mae_mlr, 'r2': r2_mlr}

# ===== MODEL 3: POLYNOMIAL REGRESSION =====
print("\n" + "="*50)
print("3️⃣  POLYNOMIAL REGRESSION (Degree 2)")
print("="*50)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

mse_poly = mean_squared_error(y_test, y_pred_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)

print(f"Mean Squared Error (MSE): {mse_poly:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_poly:.4f}")
print(f"Mean Absolute Error (MAE): {mae_poly:.4f}")
print(f"R² Score: {r2_poly:.4f}")
results['Polynomial Regression'] = {'accuracy': r2_poly, 'mse': mse_poly, 'mae': mae_poly, 'r2': r2_poly}

# ========== VISUALIZATIONS ==========
print("\n" + "="*50)
print("📊 GENERATING VISUALIZATIONS...")
print("="*50)

# Prepare data for comparison charts
models_list = list(results.keys())
accuracies = [results[m]['accuracy'] for m in models_list]
mses = [results[m]['mse'] for m in models_list]
maes = [results[m]['mae'] for m in models_list]

# 1. Accuracy Comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# R² / Accuracy Chart
axes[0].bar(models_list, accuracies, color='skyblue', edgecolor='navy', linewidth=2)
axes[0].set_ylabel('R² / Accuracy Score', fontsize=12, fontweight='bold')
axes[0].set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
axes[0].set_ylim(0, 1.1)
for i, v in enumerate(accuracies):
    axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)

# MSE Comparison
axes[1].bar(models_list, mses, color='lightcoral', edgecolor='darkred', linewidth=2)
axes[1].set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
axes[1].set_title('MSE Comparison (Lower is Better)', fontsize=13, fontweight='bold')
for i, v in enumerate(mses):
    axes[1].text(i, v + 0.001, f'{v:.4f}', ha='center', fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)

# MAE Comparison
axes[2].bar(models_list, maes, color='lightgreen', edgecolor='darkgreen', linewidth=2)
axes[2].set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
axes[2].set_title('MAE Comparison (Lower is Better)', fontsize=13, fontweight='bold')
for i, v in enumerate(maes):
    axes[2].text(i, v + 0.001, f'{v:.4f}', ha='center', fontweight='bold')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Predictions Comparison (Actual vs Predicted for all models)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Logistic Regression - Binary predictions
axes[0].scatter(y_test_binary, y_pred_log, alpha=0.6, s=50, color='blue', edgecolors='black', linewidth=0.5)
axes[0].plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Values', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Predicted Values', fontsize=11, fontweight='bold')
axes[0].set_title(f'Logistic Regression\n(Accuracy: {results["Logistic Regression"]["accuracy"]:.4f})', 
                 fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Multiple Linear Regression
axes[1].scatter(y_test, y_pred_mlr, alpha=0.6, s=50, color='green', edgecolors='black', linewidth=0.5)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Values', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Predicted Values', fontsize=11, fontweight='bold')
axes[1].set_title(f'Multiple Linear Regression\n(R²: {results["Multiple Linear Regression"]["r2"]:.4f})', 
                 fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Polynomial Regression
axes[2].scatter(y_test, y_pred_poly, alpha=0.6, s=50, color='purple', edgecolors='black', linewidth=0.5)
axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
axes[2].set_xlabel('Actual Values', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Predicted Values', fontsize=11, fontweight='bold')
axes[2].set_title(f'Polynomial Regression\n(R²: {results["Polynomial Regression"]["r2"]:.4f})', 
                 fontsize=12, fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('predictions_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Metrics Heatmap
metrics_df = pd.DataFrame({
    'Performance': accuracies,
    'MSE': mses,
    'MAE': maes
}, index=models_list)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(metrics_df, annot=True, fmt='.4f', cmap='RdYlGn', cbar_kws={'label': 'Score'}, 
            linewidths=2, linecolor='black', ax=ax)
ax.set_title('Model Performance Metrics Heatmap', fontsize=14, fontweight='bold', pad=20)
ax.set_ylabel('Models', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('metrics_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ ALL VISUALIZATIONS SAVED:")
print("   - model_comparison.png (Accuracy, MSE, MAE comparison)")
print("   - predictions_comparison.png (Actual vs Predicted)")
print("   - metrics_heatmap.png (Heatmap of all metrics)")

# Best Model Summary
best_model = max(results, key=lambda x: results[x]['accuracy'])
print(f"\n🏆 BEST MODEL: {best_model}")
print(f"   Accuracy: {results[best_model]['accuracy']:.4f}")
print(f"   MSE: {results[best_model]['mse']:.4f}")
print(f"   MAE: {results[best_model]['mae']:.4f}")