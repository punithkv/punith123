import os, sys, joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
artifacts_dir = os.path.join(BASE_DIR, "artifacts")
model = joblib.load(os.path.join(artifacts_dir, "model.joblib"))
vect  = joblib.load(os.path.join(artifacts_dir, "vectorizer.joblib"))

def predict(text):
    x = vect.transform([text])
    return model.predict(x)[0]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"message to classify\"")
        sys.exit(1)
    text = sys.argv[1]
    print("Prediction:", predict(text))
