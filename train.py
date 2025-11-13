import os
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "spam.csv")

# Load and normalize columns
df = pd.read_csv(csv_path, encoding="latin-1")[['v1', 'v2']]
df.columns = ['label', 'text']

# Inspect dataset
print("Total rows:", len(df))
print("Label distribution:")
print(df['label'].value_counts())

# Decide whether to stratify
min_count = df['label'].value_counts().min()
n_classes = df['label'].nunique()
if min_count < 2:
    # Not enough samples per class to guarantee at least one per class in test set
    print("WARNING: Some class has <2 examples. Disabling stratify for train/test split.")
    stratify_param = None
else:
    stratify_param = df['label']

# Use a reasonable test_size; keep it float if dataset is large.
test_size = 0.2

# If dataset is tiny, ensure test set size (as integer) >= n_classes when stratifying
if stratify_param is not None:
    n_samples = len(df)
    test_count = math.ceil(test_size * n_samples)
    if test_count < n_classes:
        # increase test_count to number of classes (so every class can appear in test)
        test_count = n_classes
    # use integer test_size
    test_size_for_split = test_count
    print(f"Using integer test_size={test_size_for_split} (to ensure >= {n_classes} classes in test)")
else:
    test_size_for_split = test_size

# Perform split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'],
    test_size=test_size_for_split,
    random_state=42,
    stratify=stratify_param
)

# Vectorize and train
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
