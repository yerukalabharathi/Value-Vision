import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv(r"C:\Users\birar\OneDrive\Desktop\SRM\ML PROJECT )\Chennai (1).csv")

# Select relevant numeric features for regression
numeric_features = ['Area', 'No. of Bedrooms', 'Resale', 'MaintenanceStaff', 'Gymnasium', 
                    'SwimmingPool', 'LandscapedGardens', 'JoggingTrack', 'LiftAvailable',
                    'VaastuCompliant', 'Microwave', 'GolfCourse', 'TV', 'DiningTable', 
                    'Sofa', 'Wardrobe', 'Refrigerator']

# Target variable (Price)
X = data[numeric_features]
y = data['Price']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

model_name = "Linear Regression"
print(f"{model_name}: RMSE = {rmse:f}, R-squared = {r2:f}")


# Plot
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f'Linear Regression\nRMSE: {rmse:.2f}, R2: {r2:.2f}')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()
