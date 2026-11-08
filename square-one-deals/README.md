# Square One Deals — React frontend

A four-page student capstone demo website for exploring synthetic Square One mall promotions. Uses the existing FastAPI/Hybrid NCF recommender backend.

## Setup

1. Install Node.js LTS (includes npm).
2. In this folder run `npm install`.
3. Create a folder called `public` and copy **exactly these three files** from your existing Python project into it:
   - `stores.csv`
   - `promotions_with_categories.csv`
   - `users.csv`
4. In the original Python project, activate `retail_venv` and start FastAPI: `python -m uvicorn api:app --reload`.
5. In this React project, run `npm run dev` and open the URL printed by Vite (usually http://localhost:5173).

## Pages

- `/` Home: categories, featured stores, active demo offers, personalized recommendations link
- `/stores` Stores directory with category and text filters
- `/stores/:id` Individual store's active offers (dynamic route)
- `/for-you` Recommendations fetched from the existing FastAPI `/recommendations/{user_id}` endpoint
- `/profile` Shopper and demo-date selector

## Notes

- The demo date starts at October 20, 2026 because offers are synthetic and may have historical validity dates.
- This frontend reads the three CSVs directly for store discovery and active offers. The personalized recommendations page calls FastAPI. A later phase can replace direct CSV access with dedicated API endpoints and PostgreSQL.
- No click, save, purchase, or authentication functionality is claimed. Those require backend interaction tracking.
- All offers are synthetic and the website is not affiliated with Square One Shopping Centre.
