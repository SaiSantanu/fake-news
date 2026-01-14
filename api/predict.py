from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re

app = FastAPI()

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

class News(BaseModel):
    text: str

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

@app.get("/")
def root():
    return {"status": "Fake News Detection API running"}

@app.post("/api/predict")
def predict(news: News):
    cleaned = clean_text(news.text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    return {
        "prediction": "REAL" if prediction == 1 else "FAKE"
    }
