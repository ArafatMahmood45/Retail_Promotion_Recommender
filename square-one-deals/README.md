# Square One Deals — React frontend (connected)

Uses the existing FastAPI history-aware recommender and SQLite demo interaction endpoints.

## Run

1. In your Python project, run `python -m uvicorn main:app --reload` (replace `main` with your FastAPI filename if different).
2. In this React folder, run `npm install` and `npm run dev`.
3. Open http://localhost:5173. The Vite proxy forwards `/api` requests to http://127.0.0.1:8000.
4. Select Shopper 3 and demo date 2026-10-20 under Profile; open For You.
5. Use View, Click, Save, or Simulate purchase on an offer. The action is recorded through FastAPI, and the recommendation feed refreshes.
6. Profile displays SQLite demo activity and can reset activity for the selected shopper/date.

The CSVs in `public/` are synthetic and provide store discovery. The API provides personalized rankings. No real checkout or authentication is implemented. Views have zero model preference weight. The model uses new interaction history for inference; its neural weights are not retrained on each action.
