# Step 1: Install Required Libraries

#Step 2: Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#Step 3: Load the Dataset
# Load the dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Show basic information
print(df.head())
print("\nTarget names:", data.target_names)

#Step 4: Explore the Dataset
# Check for missing values
print(df.isnull().sum())

# Visualize the target distribution
sns.countplot(x='target', data=df)
plt.title("Distribution of Benign (1) and Malignant (0) Tumors")
plt.show()

# Correlation heatmap (optional)
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

#Step 5: Split Data for Training and Testing
X = df.drop('target', axis=1)
y = df['target']

# Split into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Step 6: Standardize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Step 7: Train the SVM Classifier
svm = SVC(kernel='linear')  # You can try 'rbf', 'poly', etc.
svm.fit(X_train, y_train)

#Step 8: Evaluate the Model
y_pred = svm.predict(X_test)

# Accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Confusion matrix and classification report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))
