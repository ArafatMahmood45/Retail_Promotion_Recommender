
from contextlib import asynccontextmanager
from datetime import date as Date
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from recommend import PromotionRecommender


# Load the recommender once when the application starts.
recommender = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender
    recommender = PromotionRecommender()
    yield
    recommender = None


app = FastAPI(
    title="Retail Promotion Recommender API",
    description=(
        "Personalized promotion recommendations powered by "
        "a trained Hybrid Neural Collaborative Filtering model. "
        "All promotions are synthetic and intended for demonstration."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class PromotionResponse(BaseModel):
    promotion_id: int
    store_name: str
    promotion_category: str
    promotion_type: str
    discount_unit: str
    discount_value: float
    start_date: Date
    end_date: Date
    recommendation_score: float


class RecommendationResponse(BaseModel):
    user_id: int
    top_k: int
    as_of: Date
    mode: str
    synthetic_data: bool
    total_recommendations: int
    recommendations: list[PromotionResponse]


@app.get("/")
def home():
    return {
        "message": "Retail Promotion Recommender API",
        "status": "running",
        "model": "Hybrid NCF",
        "synthetic_data": True,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if recommender is not None else "unavailable",
        "model_loaded": recommender is not None,
    }


@app.get(
    "/recommendations/{user_id}",
    response_model=RecommendationResponse,
)
def get_recommendations(
    user_id: int,
    top_k: int = Query(default=5, ge=1, le=50),
    as_of: Optional[Date] = Query(default=None),
):
    if recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Recommendation model is unavailable.",
        )

    selected_date = as_of or Date.today()
    mode = "demo" if as_of is not None else "live"

    try:
        results = recommender.recommend(
            user_id=user_id,
            top_k=top_k,
            as_of=selected_date.isoformat(),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    recommendations = []

    for record in results.to_dict(orient="records"):
        recommendations.append(
            PromotionResponse(
                promotion_id=int(record["promotion_id"]),
                store_name=str(record["store_name"]),
                promotion_category=str(record["promotion_category"]),
                promotion_type=str(record["promotion_type"]),
                discount_unit=str(record["discount_unit"]),
                discount_value=float(record["discount_value"]),
                start_date=record["start_date"].date(),
                end_date=record["end_date"].date(),
                recommendation_score=float(record["recommendation_score"]),
            )
        )

    return RecommendationResponse(
        user_id=user_id,
        top_k=top_k,
        as_of=selected_date,
        mode=mode,
        synthetic_data=True,
        total_recommendations=len(recommendations),
        recommendations=recommendations,
    )
