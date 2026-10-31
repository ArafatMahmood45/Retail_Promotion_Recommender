# Retail Promotion Recommender

## Project Overview

Retail Promotion Recommender is an AI-powered web application designed to help shoppers discover active promotions and discounts from retail stores at Square One Shopping Centre in Mississauga, Ontario.

The application aims to bring promotions from different retailers into one platform and provide personalized recommendations based on shoppers' interactions and interests.

The current prototype uses **50 retail stores, 300 synthetic promotions, 500 synthetic users, and 32,000 generated interactions**.

## Problem Statement

Shoppers often miss discounts and limited-time promotions because information is spread across different retailer websites and platforms. This makes it difficult to discover relevant deals in one place.

Our solution brings retail promotions together in a single application and uses machine learning to personalize how promotions are presented to each shopper.

## Project Objectives

- Collect and organize retail store and promotion information.
- Generate realistic synthetic shopper interaction data.
- Develop a Hybrid Neural Collaborative Filtering (Hybrid NCF) recommendation model using PyTorch.
- Predict relevance scores for promotions based on shopper interactions and promotion characteristics.
- Build a web application for browsing stores and personalized promotions.
- Integrate a PostgreSQL database and FastAPI backend.
- Deploy the completed prototype on Microsoft Azure.

## Dataset Development

The project uses four main datasets:

| Dataset | Records | Description |
|---|---:|---|
| Users | 500 | Synthetic shopper accounts |
| Stores | 50 | Retail store names and categories |
| Promotions | 300 | Synthetic discounts, promotion types, product categories, and validity dates |
| Interactions | 32,000 | Simulated views, clicks, saves, and purchases |

### Synthetic Interaction Generation

A Python script (`generate_interactions.py`) was developed to generate shopper interactions using probabilistic rules.

The script considers shoppers' preferred stores, categories, promotion types, discount sensitivity, and exploration probability.

These preferences influence the likelihood of interacting with particular promotions while allowing shoppers to explore other categories.

The generated interactions are validated against user account creation dates and promotion validity periods.

**Note:** All promotions and shopper interactions are synthetic and do not represent verified retailer offers or actual customer behavior.

## Data Preparation

The `prepare_training_data.py` script was developed to:

1. Validate the four main datasets.
2. Combine users, stores, promotions, and interactions into `model_dataset.csv`.
3. Prepare and encode features for machine learning.
4. Split the interaction records chronologically into training, validation, and testing datasets.

### Dataset Split

| Dataset | Records | Percentage |
|---|---:|---:|
| Training | 22,400 | 70% |
| Validation | 4,800 | 15% |
| Testing | 4,800 | 15% |

The synthetic preferences used to generate interactions are excluded from model training. This allows the model to learn shopper interests from observed interaction patterns.

## Recommendation Model Development

An initial **Hybrid Neural Collaborative Filtering (Hybrid NCF)** model has been implemented using PyTorch.

The model combines:

- **Collaborative signals:** User and promotion interaction patterns.
- **Content features:** Store information, product categories, promotion types, and discount characteristics.

The model predicts engagement relevance scores that can be used to rank promotions for individual shoppers.

### Model Training

The `train_recommender.py` script:

1. Loads the prepared training and validation datasets.
2. Builds and trains the Hybrid NCF neural network.
3. Uses sampled unobserved promotions as negative training examples.
4. Evaluates the model using held-out interactions.
5. Saves the trained model as model_artifacts/hybrid_recommender.pt, along with feature mappings and evaluation results in model_artifacts/test_metrics.json.

### Model Evaluation and Baseline Comparison

The Hybrid NCF model was trained for 12 epochs and evaluated using held-out synthetic shopper interactions.

To assess its recommendation performance, a **popularity-based recommendation baseline** was implemented in `evaluate_baseline.py`.

The baseline ranks promotions according to the number of clicks, saves, and purchases recorded in the training dataset. Unlike the Hybrid NCF model, it does not learn individual shopper preferences or use promotion characteristics to predict relevance.

Both recommendation approaches were evaluated using the same 417 eligible test users, with consistent evaluation criteria:

- Only promotions active at the evaluation timestamp were considered.
- Promotions previously interacted with during training and validation were excluded.
- Clicks, saves, and purchases were treated as positive interactions.
- Recommendation performance was measured using Precision@5, Recall@5, and Hit Rate@5.

#### Model Performance Comparison

| Evaluation Metric | Hybrid NCF | Popularity Baseline |
|---|---:|---:|
| Precision@5 | **7.29%** | 5.42% |
| Recall@5 | **14.95%** | 11.92% |
| Hit Rate@5 | **31.65%** | 24.46% |

The Hybrid NCF model outperformed the popularity-based baseline across all three evaluation metrics.

These results suggest that combining shopper interaction patterns with store and promotion features can improve recommendation ranking compared with relying solely on historical promotion popularity.

The trained model, feature mappings, and evaluation results are saved in the `model_artifacts/` directory.

**Note:** All results are based on synthetic shopper interactions and do not represent validated performance with real customers.

## Technology Stack

- **Python, Pandas & NumPy:** Data generation and processing
- **PyTorch:** Hybrid recommendation model development
- **PostgreSQL:** Relational database (planned integration)
- **FastAPI:** Backend API (planned)
- **Streamlit:** Web application interface (planned)
- **Docker:** Containerization (planned)
- **Microsoft Azure:** Cloud deployment (planned)
- **Git & GitHub:** Version control and collaboration

## Project Team

- **Fabian Akaeze:** Project Manager & Retail Product Analyst
- **Kalik Reece:** AI/ML Engineer
- **Arafat Mahmood:** Data & Backend Engineer
- **Orukome Egbuwoku:** MLOps & Cloud Engineer

## Project Status

### Completed

- [x] Initialized the project repository and development environment.
- [x] Prepared 50 retail store records.
- [x] Created 300 synthetic promotions with categories and validity periods.
- [x] Generated 500 synthetic shopper profiles and preferences.
- [x] Developed Python-based synthetic interaction generation.
- [x] Generated 32,000 shopper interactions.
- [x] Combined and validated the datasets for machine learning.
- [x] Prepared training, validation, and test datasets.
- [x] Implemented and trained an initial Hybrid NCF recommendation model.
- [x] Evaluated the initial model and saved its trained parameters.
- - [x] Implemented a popularity-based recommendation baseline using historical shopper engagement.
- [x] Evaluated Hybrid NCF against the popularity baseline using consistent candidate eligibility and ranking metrics.

### Remaining Work

- [ ] Improve model performance and evaluate recommendations at different ranking sizes.
- [ ] Create and populate the PostgreSQL database.
- [ ] Develop FastAPI endpoints for stores, promotions, and personalized recommendations.
- [ ] Build the Streamlit user interface.
- [ ] Connect user interactions to the backend and recommendation system.
- [ ] Integrate and test the complete application.
- [ ] Containerize the application using Docker.
- [ ] Deploy the prototype to Microsoft Azure.
- [ ] Complete final testing, documentation, and project demonstration.

## Expected Outcome

The completed application will allow shoppers to browse active promotions from participating retail stores and receive personalized promotion rankings based on their interaction history.

The prototype will demonstrate how data engineering, machine learning, backend development, and cloud technologies can be combined to build a retail recommendation system.