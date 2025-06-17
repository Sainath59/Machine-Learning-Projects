#Step 1: Install Required Libraries

#Step 2: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#Step 3: Load the Dataset
# Load dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

print(df)
print(df.head())

#Step 4: Explore the Dataset
print(df['label'].value_counts())

# Visualize distribution
sns.countplot(data=df, x='label')
plt.title('Spam vs Ham (Not Spam)')
plt.show()

#Step 5: Preprocess the Data
# Convert labels to binary
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Split features and labels
X = df['message']
y = df['label_num']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#Step 6: Text Vectorization (Convert text to numbers)
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

#Step 7: Train the Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_vect, y_train)

#Step 8: Evaluate the Model
y_pred = model.predict(X_test_vect)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#Step 9: Test with a Custom Email
sample = ["Congratulations! You've won a free ticket to the Bahamas! Click here."]
sample_vect = vectorizer.transform(sample)
print("Spam" if model.predict(sample_vect)[0] == 1 else "Not Spam")
