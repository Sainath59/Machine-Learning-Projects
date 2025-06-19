import pandas as pd

# Sample credit data
data = {
    "Income": [50000, 60000, 30000, 80000, 20000, 40000, 70000, 100000],
    "CreditScore": [700, 720, 610, 780, 590, 650, 730, 800],
    "LoanAmount": [20000, 25000, 10000, 30000, 5000, 15000, 22000, 35000],
    "YearsEmployed": [5, 6, 2, 10, 1, 3, 7, 12],
    "Risk": ["No", "No", "Yes", "No", "Yes", "Yes", "No", "No"]
}

df = pd.DataFrame(data)
print(df)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Encode the target column
label_encoder = LabelEncoder()
df['Risk'] = label_encoder.fit_transform(df['Risk'])  # No = 0, Yes = 1

# Split data
X = df.drop("Risk", axis=1)
y = df["Risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Encode the target column
label_encoder = LabelEncoder()
df['Risk'] = label_encoder.fit_transform(df['Risk'])  # No = 0, Yes = 1

# Split data
X = df.drop("Risk", axis=1)
y = df["Risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier

# Create and train the model
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plot_tree(model, feature_names=X.columns.tolist(), class_names=["No Risk", "Risk"], filled=True)
plt.title("Decision Tree for Credit Risk Prediction")
plt.show()
