# somalia_electricity_model_real.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load the real dataset
df = pd.read_csv("somalia_electricity_data_real.csv")

# Explore the dataset
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset summary:")
print(df.describe())

# Visualize relationships
sns.pairplot(df)
plt.suptitle("Pairplot of Somalia Real Electricity Data", y=1.02)
plt.show()

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Split the data
X = df_encoded.drop("Electricity_kWh_Price_USD", axis=1)
y = df_encoded["Electricity_kWh_Price_USD"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("\nLinear Regression Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("RMSE:", mean_squared_error(y_test, y_pred_lr, squared=False))
print("R²:", r2_score(y_test, y_pred_lr))

# Train Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("RMSE:", mean_squared_error(y_test, y_pred_rf, squared=False))
print("R²:", r2_score(y_test, y_pred_rf))

# Feature importance plot
importances = rf.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
plt.title("Top 10 Feature Importances - Real Data")
plt.tight_layout()
plt.show()

# Save model
joblib.dump(rf, "electricity_rf_model_real.pkl")

# Predict example
sample = X_test.iloc[0].values.reshape(1, -1)
prediction = rf.predict(sample)
print("\nPredicted Price for a Sample Region:", round(prediction[0], 2))
