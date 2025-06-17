#Step 1: Install Required Libraries

#Step 2: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 3: Load the Dataset
# Load data from a CSV file
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')
print(df.head())
print(df.describe())

#Step 4: Visualize the Data
# Correlation matrix
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()
# Pairplot (optional)
# sns.pairplot(df)
# plt.show()

#Step 5: Preprocess the Data
# Features and target
X = df.drop('medv', axis=1)  # 'medv' is the target (Median home value)
y = df['medv']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Step 6: Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

#Step 7: Evaluate the Model
# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Visualize Predictions
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

#Step 8: Predict on New Data
# Example: Predict for new input (13 feature values)
new_data = [[0.2, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296.0, 15.3, 396.9, 4.98]]
prediction = model.predict(new_data)
print("Predicted Price:", prediction[0])
