from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Annotated
from pathlib import Path
import pickle
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import threading
import webbrowser
import uvicorn

# -----------------------
# NLTK setup
# -----------------------
nltk.download("punkt")
nltk.download("stopwords")
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def transform_text(text: str) -> str:
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w.isalnum()]
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)


# -----------------------
# Load ML models & data
# -----------------------
BASE_DIR = Path(__file__).resolve().parent

model1 = pickle.load(open(BASE_DIR / "model12.pkl", "rb"))
vectorizer = pickle.load(open(BASE_DIR / "vectorizer.pkl", "rb"))

popular_df = pickle.load(open(BASE_DIR / "popular.pkl", "rb"))
pt = pickle.load(open(BASE_DIR / "pt.pkl", "rb"))
books = pickle.load(open(BASE_DIR / "books.pkl", "rb"))
similarity_score = pickle.load(open(BASE_DIR / "similarity_score.pkl", "rb"))


# -----------------------
# FastAPI setup
# -----------------------
app = FastAPI(title="Combined App", version="1.1")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# -----------------------
# Data Models
# -----------------------
class UserInput(BaseModel):
    sms: Annotated[str, Field(..., description="Enter SMS message")]


# -----------------------
# Health Check Route
# -----------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}


# -----------------------
# Book Recommendation Routes
# -----------------------
@app.get("/book", response_class=HTMLResponse)
def index(request: Request):
    books_data = [
        {
            "book_name": str(row["Book-Title"]),
            "author": str(row["Book-Author"]),
            "image": str(row["Image-URL-M"]),
            "votes": int(row["num_rating"]),
            "rating": float(row["avg_rating"]),
        }
        for _, row in popular_df.iterrows()
    ]
    return templates.TemplateResponse("index.html", {"request": request, "books_data": books_data})


@app.get("/recommend", response_class=HTMLResponse)
def recommend_ui(request: Request):
    return templates.TemplateResponse("recommend.html", {"request": request})


@app.post("/recommend_books", response_class=HTMLResponse)
def recommend_books(request: Request, user_input: str = Form(...)):
    user_input = user_input.strip()
    if user_input not in pt.index:
        raise HTTPException(status_code=404, detail=f"Book '{user_input}' not found")

    index = np.where(pt.index == user_input)[0][0]

    similar_items = sorted(
        list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True
    )[1:6]

    recommended_books = []
    for i in similar_items:
        temp_df = books[books["Book-Title"] == pt.index[i[0]]].drop_duplicates("Book-Title")
        recommended_books.append(
            {
                "book_name": str(temp_df["Book-Title"].values[0]),
                "author": str(temp_df["Book-Author"].values[0]),
                "image": str(temp_df["Image-URL-M"].values[0]),
            }
        )

    return templates.TemplateResponse(
        "recommend.html",
        {"request": request, "recommended_books": recommended_books, "user_input": user_input},
    )


# -----------------------
# SMS Spam Classifier Routes
# -----------------------
@app.get("/sms", response_class=HTMLResponse)
def serve_frontend():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>SMS Spam Classifier</title>
        <style>
            body { font-family: Arial; background: #f7f7f7; display: flex; justify-content: center; align-items: center; height: 100vh; }
            .container { background: white; padding: 20px; border-radius: 10px; box-shadow: 0px 0px 10px rgba(0,0,0,0.1); width: 400px; }
            textarea { width: 100%; height: 100px; padding: 10px; margin-bottom: 10px; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; border-radius: 5px; }
            button:hover { background: #0056b3; }
            .result { margin-top: 15px; font-weight: bold; }
        </style>
    </head>
    <body>
    <div class="container">
        <h2>SMS Spam Classifier</h2>
        <textarea id="smsInput" placeholder="Enter your SMS here..."></textarea>
        <button onclick="predictSMS()">Check</button>
        <div id="result" class="result"></div>
    </div>
    <script>
        async function predictSMS() {
            const sms = document.getElementById("smsInput").value;
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ sms: sms })
            });
            if (response.ok) {
                const data = await response.json();
                document.getElementById("result").innerHTML = `Prediction: ${data.prediction}`;
            } else {
                document.getElementById("result").innerHTML = "Error: Unable to get prediction.";
            }
        }
    </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/predict")
def predict_sms(data: UserInput):
    try:
        transformed_sms = transform_text(data.sms)
        vector_input = vectorizer.transform([transformed_sms])
        result1 = model1.predict(vector_input)[0]  # no .toarray()
        prediction_label = "spam" if result1 == 1 else "not spam"
        return JSONResponse(status_code=200, content={"prediction": prediction_label})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# -----------------------
# Launch frontend automatically
# -----------------------
def open_frontend():
    webbrowser.open("http://127.0.0.1:8000/")


if __name__ == "__main__":
    threading.Timer(1.5, open_frontend).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
