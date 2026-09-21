# Spam Detector

Simple spam detector using MultinomialNB and CountVectorizer.

## Files
- train.py — trains model and saves artifacts
- predict.py — CLI prediction
- app.py — Flask demo
- spam.csv — dataset (already in repo)

## Run locally
1. python -m venv env
2. source env/Scripts/activate   # Git Bash on Windows
3. pip install -r requirements.txt
4. python train.py
5. python app.py
