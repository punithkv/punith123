from flask import Flask, request, render_template_string
import os, joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
artifacts_dir = os.path.join(BASE_DIR, "artifacts")
model = joblib.load(os.path.join(artifacts_dir, "model.joblib"))
vect  = joblib.load(os.path.join(artifacts_dir, "vectorizer.joblib"))

app = Flask(__name__)

TEMPLATE = """
<html><body style="font-family:Arial,Helvetica,sans-serif">
  <h2>Spam Detector</h2>
  <form method="post">
    <textarea name="text" rows="6" cols="60">{{ text }}</textarea><br><br>
    <button type="submit">Predict</button>
  </form>
  {% if label %}
  <h3>Prediction: {{ label }}</h3>
  {% endif %}
</body></html>
"""

@app.route("/", methods=["GET","POST"])
def index():
    label = None
    text = ""
    if request.method == "POST":
        text = request.form.get("text","")
        x = vect.transform([text])
        label = model.predict(x)[0]
    return render_template_string(TEMPLATE, label=label, text=text)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
