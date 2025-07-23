# -----------------------------
# 🔹 Importing Required Libraries
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge

# -----------------------------
# 🔹 Load and Explore the Dataset
# -----------------------------
data = pd.read_csv("Overview.csv", encoding="latin1")
print("Data Shape:", data.shape)
print("Missing Values:\n", data.isnull().sum())

# -----------------------------
# 🔹 Data Cleaning
# -----------------------------
# Fill missing numeric values with column mean
data.fillna(data.mean(numeric_only=True), inplace=True)

# Fill categorical columns with mode
for col in data.select_dtypes(include=['object']).columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Drop unnecessary column if exists
if 'Unnamed: 13' in data.columns:
    data.drop(columns='Unnamed: 13', inplace=True)

# -----------------------------
# 🔹 Data Visualization
# -----------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Battery Temperature (End)', y='Battery Temperature (Start) [°C]', data=data, alpha=0.6)
plt.title("Battery Temperature (End) vs. Start")
plt.xlabel("Battery Temperature (End)")
plt.ylabel("Battery Temperature (Start) [°C]")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Violin Plot for Distribution
plt.figure(figsize=(15, 6))
sns.violinplot(data=data)
plt.title("Violin Plot of Features")
plt.show()

# -----------------------------
# 🔹 Encoding Categorical Variables
# -----------------------------
le = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = le.fit_transform(data[col])

# -----------------------------
# 🔹 Feature Engineering
# -----------------------------
X = data.drop(['Battery Temperature (End)'], axis=1)
y = data['Battery Temperature (End)']

# -----------------------------
# 🔹 Data Splitting
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 🔹 Metric Evaluation Function
# -----------------------------
def calculatemetrics(algorithm, predict, testY):
    mse = mean_squared_error(testY, predict)
    mae = mean_absolute_error(testY, predict)
    r2 = r2_score(testY, predict) * 100

    print(f"{algorithm} - MSE: {mse}")
    print(f"{algorithm} - MAE: {mae}")
    print(f"{algorithm} - R² Score: {r2}%")

    # Scatter Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(testY, predict, color='blue', alpha=0.5)
    plt.plot([np.min(testY), np.max(testY)], [np.min(testY), np.max(testY)], linestyle='--', color='red')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f"{algorithm} - Predicted vs Actual")
    plt.grid(True)
    plt.show()

# -----------------------------
# 🔹 Linear Regression with Noise
# -----------------------------
noise_factor = 0.9
X_train_noisy = X_train + np.random.normal(0, noise_factor, X_train.shape)
X_test_noisy = X_test + np.random.normal(0, noise_factor, X_test.shape)

lr_model = LinearRegression()
lr_model.fit(X_train_noisy, y_train)
y_pred_lr = lr_model.predict(X_test_noisy)

calculatemetrics("Linear Regression with Noise", y_pred_lr, y_test)

# -----------------------------
# 🔹 Ridge Regression (Regularized)
# -----------------------------
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

calculatemetrics("Ridge Regression", y_pred_ridge, y_test)

# -----------------------------
# 🔹 Predicting on New Test Data
# -----------------------------
# Load test dataset and preprocess (drop target column if present)
test = pd.read_csv('testdata.csv')
if 'Battery Temperature (End)' in test.columns:
    test = test.drop(['Battery Temperature (End)'], axis=1)

# Label Encoding for consistency
for col in test.select_dtypes(include=['object']).columns:
    test[col] = le.fit_transform(test[col])

# Predict using trained Ridge model
pred = ridge_model.predict(test)
test['prediction'] = pred

# Save the prediction results
test.to_csv("predicted_results.csv", index=False)
print("✅ Prediction completed and saved to 'predicted_results.csv'")
