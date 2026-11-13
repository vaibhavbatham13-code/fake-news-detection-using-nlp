"""
FastAPI service for fake news detection.

Run locally:
    uvicorn api:app --reload

Then open http://127.0.0.1:8000/docs to try the API in your browser.

Example request:
    curl -X POST http://127.0.0.1:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"title": "Headline", "text": "Article body with at least 100 characters..."}'
"""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from utilis import clean_text, explain, fake_probability, verdict

MODEL_PATH = Path(__file__).resolve().parent / "model" / "fake_news_model.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the server starts, not on every request."""
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run train_model.py first.")
    app.state.model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(title="Fake News Detector API", version="1.0.0", lifespan=lifespan)


class ArticleIn(BaseModel):
    title: str = Field("", max_length=500)
    text: str = Field(..., min_length=100, max_length=50_000)


class PredictionOut(BaseModel):
    fake_score: float = Field(..., description="0 = very likely real, 1 = very likely fake")
    verdict: str = Field(..., description="'fake', 'real' or 'uncertain'")
    words_toward_fake: list[str]
    words_toward_real: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOut)
def predict(article: ArticleIn):
    try:
        combined = f"{article.title} {article.text}".strip()

        if not clean_text(combined):
            raise HTTPException(
                status_code=422,
                detail="No usable words after cleaning the text."
            )

        model = app.state.model
        score = fake_probability(model, combined)
        fake_terms, real_terms = explain(model, combined)

        return PredictionOut(
            fake_score=round(score, 4),
            verdict=verdict(score),
            words_toward_fake=[term for term, _ in fake_terms],
            words_toward_real=[term for term, _ in real_terms],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {e}"
        )


