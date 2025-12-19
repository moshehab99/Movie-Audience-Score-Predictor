Movie Audience Score Predictor 🎬📊
A machine learning project that predicts the Audience Score (TMDB vote_average) for movies using the popular TMDB 5000 Movies dataset.
📌 Project Overview
This project demonstrates a complete end-to-end machine learning pipeline for regression tasks:

Data Cleaning & Preprocessing
Exploratory Data Analysis (EDA) with interactive visualizations
Feature Engineering (text + numerical features)
Model Training with Random Forest and hyperparameter tuning
Interactive Deployment via a Streamlit web app

The goal is to predict how well a movie will be received by audiences (on a scale from 0 to 10) based on metadata like budget, overview, popularity, cast size, genres, keywords, and release date.
📊 Dataset

Source: TMDB 5000 Movies Dataset on Kaggle
Two main files:
tmdb_5000_movies.csv → Movie metadata (budget, revenue, genres, overview, etc.)
tmdb_5000_credits.csv → Cast and crew information

Final processed dataset contains ~4800 movies after cleaning.

🛠️ Key Features Engineered

Text Features:
Cleaned movie overview (lowercase, remove punctuation/numbers, stopwords, lemmatization)
TF-IDF vectorization on overview (top 5000 features)
Overview length as a numerical feature

Numerical Features:
Budget, Revenue, Runtime, Popularity, Vote Count
Release Year & Month (extracted from release date)
Number of genres, keywords, and main cast members

All numerical features scaled using StandardScaler

Note: Instead of one-hot encoding genres/keywords (which creates high dimensionality), we used count-based features (number of genres/keywords) for better performance and simplicity.
📈 Exploratory Data Analysis (EDA)
Interactive visualizations using Plotly Express covering:

Distribution of audience scores
Score trends over time (by release year)
Most common movie genres
Relationship between budget/revenue and audience score
Impact of popularity & vote count
Seasonal patterns (score by release month)
Correlation heatmap of numerical features

🤖 Model

Algorithm: Random Forest Regressor
Tuning: GridSearchCV for optimal hyperparameters
Best Model Performance (on test set):
MAE: ~0.53
RMSE: ~0.75
R²: ~0.52

Strongest predictors: popularity, vote_count, revenue, and TF-IDF text features from overview

🚀 Deployment
A simple and interactive Streamlit web app (app.py) allows users to:

Input movie details (overview, budget, cast size, etc.)
Get real-time predicted audience score
No external dependencies beyond saved model artifacts

Run locally with:
Bashstreamlit run app.py
📁 Project Structure
text.
├── Audience_Score.ipynb              # Main Jupyter notebook (EDA + modeling)
├── app.py                            # Streamlit prediction app
├── best_rf_model.pkl                 # Trained Random Forest model
├── tfidf_vectorizer.pkl              # Fitted TF-IDF vectorizer
├── scaler.pkl                        # Fitted StandardScaler
├── tmdb_5000_movies.csv              # Raw data
├── tmdb_5000_credits.csv             # Raw data
└── README.md                         # This file
🛠️ Requirements

Python 3.8+
Libraries: pandas, numpy, scikit-learn, plotly, streamlit, nltk, scipy

Install dependencies:
Bashpip install pandas numpy scikit-learn plotly streamlit scipy
🎯 Future Improvements

Try advanced models (XGBoost, LightGBM, or neural networks with GRU/LSTM on text)
Add genre/keyword one-hot encoding or embeddings
Incorporate poster images or trailer sentiment
Deploy publicly (e.g., on Streamlit Community Cloud or Hugging Face Spaces)


Built for learning and portfolio purposes – feel free to fork, improve, or use as inspiration! 🚀
Made by moShehab
