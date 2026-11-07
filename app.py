
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

BASE = Path(__file__).resolve().parent
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Retail Promotion Recommender",
    page_icon="🛍️",
    layout="wide",
)

@st.cache_data
def load_data():
    users = pd.read_csv(BASE / "users.csv")
    stores = pd.read_csv(BASE / "stores.csv")
    promotions = pd.read_csv(
        BASE / "promotions_with_categories.csv",
        parse_dates=["start_date", "end_date"],
    )
    return users, stores, promotions


def format_discount(unit, value, promotion_type):
    value = float(value)

    if unit == "percent":
        amount = f"{value:g}%"
    elif unit == "CAD":
        amount = f"${value:g}"
    else:
        amount = f"{value:g} {unit}"

    if promotion_type == "second_item_percentage_off":
        return f"{amount} off second item"

    return f"{amount} off"


def get_recommendations(user_id, selected_date, top_k):
    response = requests.get(
        f"{API_URL}/recommendations/{user_id}",
        params={
            "top_k": top_k,
            "as_of": selected_date.isoformat(),
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


users, stores, promotions = load_data()

st.title("🛍️ Retail Promotion Recommender")
st.caption("Personalized shopping promotions powered by Hybrid NCF")

st.info(
    "Demo application: All shoppers and promotions are synthetic. "
    "Offers displayed here are not verified Square One discounts."
)

with st.sidebar:
    st.header("Shopper Settings")

    user_id = st.selectbox(
        "Select Shopper",
        options=sorted(users["user_id"].astype(int).tolist()),
        format_func=lambda value: f"Shopper {value}",
    )

    selected_date = st.date_input(
        "Promotion Date (Demo)",
        value=date(2026, 10, 20),
    )

    top_k = st.slider(
        "Number of Recommendations",
        min_value=1,
        max_value=20,
        value=5,
    )

    st.divider()
    st.caption("Recommendation Model: Hybrid NCF")
    st.caption("Backend: FastAPI")
    st.caption("Frontend: Streamlit")


recommendations_tab, browse_tab = st.tabs(
    ["Personalized Recommendations", "Browse Promotions"]
)

with recommendations_tab:
    st.subheader(f"Recommended Promotions for Shopper {user_id}")
    st.write(f"Showing promotions valid on {selected_date:%B %d, %Y}.")

    try:
        result = get_recommendations(
            user_id=user_id,
            selected_date=selected_date,
            top_k=top_k,
        )

        recommendations = result["recommendations"]

        if not recommendations:
            st.warning(
                "No eligible promotions were found for this shopper "
                "on the selected date."
            )
        else:
            st.success(
                f"Found {len(recommendations)} personalized recommendations."
            )

            for rank, promo in enumerate(recommendations, start=1):
                with st.container(border=True):
                    left, right = st.columns([3, 1])

                    with left:
                        st.markdown(
                            f"### {rank}. {promo['store_name']}"
                        )
                        st.write(
                            f"**Category:** {promo['promotion_category']}"
                        )
                        st.write(
                            "**Offer:** "
                            + format_discount(
                                promo["discount_unit"],
                                promo["discount_value"],
                                promo["promotion_type"],
                            )
                        )
                        st.caption(
                            f"Valid: {promo['start_date']} "
                            f"to {promo['end_date']}"
                        )

                    with right:
                        st.metric(
                            "Model Score",
                            f"{promo['recommendation_score']:.3f}",
                        )
                        st.caption(
                            f"Promotion ID: {promo['promotion_id']}"
                        )

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to FastAPI. "
            "Start the backend with: "
            "python -m uvicorn api:app --reload"
        )
    except requests.exceptions.RequestException as exc:
        st.error(f"API request failed: {exc}")


with browse_tab:
    st.subheader("Browse Active Promotions")

    active = promotions.loc[
        (promotions["start_date"].dt.date <= selected_date)
        & (promotions["end_date"].dt.date >= selected_date)
    ].copy()

    active = active.merge(
        stores[["store_id", "store_name", "category"]],
        on="store_id",
        how="left",
    )

    categories = sorted(active["category"].dropna().unique().tolist())

    selected_category = st.selectbox(
        "Filter by Store Category",
        options=["All Categories"] + categories,
    )

    if selected_category != "All Categories":
        active = active.loc[
            active["category"] == selected_category
        ]

    st.write(f"**{len(active)} active promotions found.**")

    if active.empty:
        st.info("No promotions match the selected date and category.")
    else:
        active["offer"] = active.apply(
            lambda row: format_discount(
                row["discount_unit"],
                row["discount_value"],
                row["promotion_type"],
            ),
            axis=1,
        )

        st.dataframe(
            active[
                [
                    "store_name",
                    "category",
                    "promotion_category",
                    "offer",
                    "start_date",
                    "end_date",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
