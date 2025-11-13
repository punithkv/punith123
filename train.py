import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "spam.csv")

# Load
df = pd.read_csv(csv_path, encoding="latin-1")[['v1', 'v2']]
df.columns = ['label', 'text']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Vectorize + model
vect = CountVectorizer(stop_words='english')
X_train_vec = vect.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate
X_test_vec = vect.transform(X_test)
pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# Save artifacts
artifacts_dir = os.path.join(BASE_DIR, "artifacts")
os.makedirs(artifacts_dir, exist_ok=True)
joblib.dump(model, os.path.join(artifacts_dir, "model.joblib"))
joblib.dump(vect, os.path.join(artifacts_dir, "vectorizer.joblib"))
print("Saved model and vectorizer to", artifacts_dir)
