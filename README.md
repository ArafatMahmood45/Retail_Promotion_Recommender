# Retail Promotion Recommender

## Project Overview

Retail Promotion Recommender is an AI-powered web application designed to help shoppers discover active promotions and discounts from retail stores at Square One Shopping Centre in Mississauga, Ontario.

The application will allow users to explore available promotions across approximately 59 stores and receive personalized recommendations based on their interactions with stores and promotions.

## Problem Statement

Shoppers often miss discounts and limited-time promotions because information is spread across different retailer websites and platforms. This makes it difficult to discover relevant deals in one place.

Our solution will bring retail promotions together in a single application and use machine learning to personalize how promotions are presented to each shopper.

## Project Objectives

- Collect and organize retail store and promotion information.
- Generate synthetic shopper interaction data for model development.
- Develop a neural-network-based hybrid recommendation model using PyTorch.
- Predict relevance scores for active promotions based on shopper interactions and promotion characteristics.
- Build a web application for browsing stores and personalized promotions.
- Deploy the completed prototype on Microsoft Azure.

## How the Recommendation System Works

The system will use four connected datasets:

1. **Users:** Synthetic shoppers with different interests and preferences.
2. **Stores:** Retail stores, including names and categories.
3. **Promotions:** Available discounts, promotion types, and validity dates.
4. **Interactions:** Shopper activities such as views, clicks, saves, and purchases.

The recommendation model will learn from historical interactions and store/promotion characteristics to predict the relevance of active promotions for individual users.

Promotions will be ranked based on their predicted relevance, while all active promotions remain accessible.

## Technology Stack

- **Python & Pandas:** Data processing and preparation
- **PostgreSQL:** Relational database
- **PyTorch:** Neural-network model development
- **FastAPI:** Backend API
- **Streamlit:** Web application interface
- **Docker:** Containerization
- **Microsoft Azure:** Cloud deployment
- **GitHub:** Version control and team collaboration

## Project Team

- **Fabian Akaeze:** Project Manager & Retail Product Analyst
- **Kalik Reece:** AI/ML Engineer
- **Arafat Mahmood:** Data & Backend Engineer
- **Orukome Egbuwoku:** MLOps & Cloud Engineer

## Project Status

Currently in the initial development phase, focusing on project setup, retail data collection, database design, and preparation for recommendation model development.