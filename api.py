
from contextlib import asynccontextmanager
from datetime import date as Date, datetime, time
from pathlib import Path
from typing import Literal, Optional
import sqlite3

import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from recommend import PromotionRecommender


BASE = Path(__file__).resolve().parent
DATABASE = BASE / "demo_interactions.db"

recommender = None


def get_connection():
    connection = sqlite3.connect(
        DATABASE,
        timeout=10,
    )
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                promotion_id INTEGER NOT NULL,
                interaction_type TEXT NOT NULL
                    CHECK (
                        interaction_type IN (
                            'view', 'click', 'save', 'purchase'
                        )
                    ),
                interaction_timestamp TEXT NOT NULL,
                recorded_at TEXT NOT NULL
                    DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_user_interactions
            ON interactions(user_id, interaction_timestamp)
            """
        )


def read_interactions(user_id, as_of=None):
    query = """
        SELECT
            id,
            user_id,
            promotion_id,
            interaction_type,
            interaction_timestamp
        FROM interactions
        WHERE user_id = ?
    """

    parameters = [int(user_id)]

    if as_of is not None:
        query += " AND substr(interaction_timestamp, 1, 10) = ?"
        parameters.append(as_of.isoformat())

    query += " ORDER BY interaction_timestamp, id"

    with get_connection() as connection:
        rows = connection.execute(
            query, parameters
        ).fetchall()

    return [dict(row) for row in rows]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender

    initialize_database()
    recommender = PromotionRecommender()

    yield

    recommender = None


app = FastAPI(
    title="Retail Promotion Recommender API",
    description=(
        "History-Aware Hybrid Neural Recommendation Model "
        "with persistent demo interaction tracking. "
        "All user and promotion data are synthetic."
    ),
    version="3.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


class InteractionRequest(BaseModel):
    user_id: int = Field(gt=0)
    promotion_id: int = Field(gt=0)
    interaction_type: Literal[
        "view", "click", "save", "purchase"
    ]
    as_of: Date


class InteractionResponse(BaseModel):
    id: int
    user_id: int
    promotion_id: int
    interaction_type: str
    interaction_timestamp: str
    message: str


def require_model():
    if recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Recommendation model is unavailable.",
        )
    return recommender


@app.get("/")
def home():
    return {
        "message": "Retail Promotion Recommender API",
        "status": "running",
        "model": "History-Aware Hybrid Neural Recommender",
        "synthetic_data": True,
        "interaction_storage": "SQLite",
    }


@app.get("/health")
def health():
    return {
        "status": (
            "healthy" if recommender is not None
            else "unavailable"
        ),
        "model_loaded": recommender is not None,
    }


@app.get(
    "/recommendations/{user_id}",
    response_model=RecommendationResponse,
)
def get_recommendations(
    user_id: int,
    top_k: int = Query(default=10, ge=1, le=50),
    as_of: Optional[Date] = Query(default=None),
):
    model = require_model()

    selected_date = as_of or Date.today()

    # Read new interactions saved in SQLite.
    new_events = read_interactions(
        user_id,
        as_of=selected_date,
    )

    extra = (
        pd.DataFrame(new_events)
        if new_events
        else None
    )

    try:
        results = model.recommend(
            user_id=user_id,
            top_k=top_k,
            as_of=selected_date.isoformat(),
            extra_interactions=extra,
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
                promotion_category=str(
                    record["promotion_category"]
                ),
                promotion_type=str(
                    record["promotion_type"]
                ),
                discount_unit=str(
                    record["discount_unit"]
                ),
                discount_value=float(
                    record["discount_value"]
                ),
                start_date=record["start_date"].date(),
                end_date=record["end_date"].date(),
                recommendation_score=float(
                    record["recommendation_score"]
                ),
            )
        )

    return RecommendationResponse(
        user_id=user_id,
        top_k=top_k,
        as_of=selected_date,
        mode="demo" if as_of is not None else "live",
        synthetic_data=True,
        total_recommendations=len(recommendations),
        recommendations=recommendations,
    )


@app.post(
    "/interactions",
    response_model=InteractionResponse,
    status_code=201,
)
def record_interaction(event: InteractionRequest):
    model = require_model()

    if not model.valid_user(event.user_id):
        raise HTTPException(
            status_code=404,
            detail="User does not exist.",
        )

    if not model.valid_promotion(
        event.promotion_id,
        event.as_of.isoformat(),
    ):
        raise HTTPException(
            status_code=400,
            detail="Promotion is not active on the selected date.",
        )

    # Demo events use the selected demo date.
    # recorded_at separately captures actual insertion time.
    timestamp = datetime.combine(
        event.as_of,
        time(hour=12),
    ).isoformat(timespec="seconds")

    with get_connection() as connection:
        # A purchase can only be recorded once per user,
        # promotion, and demo date.
        if event.interaction_type == "purchase":
            existing = connection.execute(
                """
                SELECT id
                FROM interactions
                WHERE user_id = ?
                  AND promotion_id = ?
                  AND interaction_type = 'purchase'
                  AND substr(interaction_timestamp, 1, 10) = ?
                """,
                (
                    event.user_id,
                    event.promotion_id,
                    event.as_of.isoformat(),
                ),
            ).fetchone()

            if existing:
                raise HTTPException(
                    status_code=409,
                    detail="This promotion was already purchased.",
                )

        cursor = connection.execute(
            """
            INSERT INTO interactions (
                user_id,
                promotion_id,
                interaction_type,
                interaction_timestamp
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                event.user_id,
                event.promotion_id,
                event.interaction_type,
                timestamp,
            ),
        )

        interaction_id = cursor.lastrowid

    return InteractionResponse(
        id=interaction_id,
        user_id=event.user_id,
        promotion_id=event.promotion_id,
        interaction_type=event.interaction_type,
        interaction_timestamp=timestamp,
        message="Interaction recorded successfully.",
    )


@app.get("/interactions/{user_id}")
def get_user_interactions(
    user_id: int,
    as_of: Optional[Date] = Query(default=None),
):
    model = require_model()

    if not model.valid_user(user_id):
        raise HTTPException(
            status_code=404,
            detail="User does not exist.",
        )

    events = read_interactions(user_id, as_of)

    return {
        "user_id": user_id,
        "total_interactions": len(events),
        "interactions": events,
    }


@app.delete("/interactions/{user_id}")
def reset_user_interactions(
    user_id: int,
    as_of: Optional[Date] = Query(default=None),
):
    model = require_model()

    if not model.valid_user(user_id):
        raise HTTPException(
            status_code=404,
            detail="User does not exist.",
        )

    query = "DELETE FROM interactions WHERE user_id = ?"
    parameters = [user_id]

    if as_of is not None:
        query += (
            " AND substr(interaction_timestamp, 1, 10) = ?"
        )
        parameters.append(as_of.isoformat())

    with get_connection() as connection:
        cursor = connection.execute(
            query,
            parameters,
        )
        deleted = cursor.rowcount

    return {
        "message": "Demo activity reset successfully.",
        "user_id": user_id,
        "deleted_interactions": deleted,
    }
