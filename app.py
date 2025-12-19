import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# تحميل المكونات (بس اللي موجودين فعلاً)
@st.cache_resource
def load_artifacts():
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('best_rf_model.pkl')
    return tfidf, scaler, model

tfidf, scaler, model = load_artifacts()

# نفس دالة التنظيف اللي في النوت بوك
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# الواجهة
st.title("🎥 Movie Audience Score Predictor")
st.markdown("Enter the movie details so we can predict the Audience Score (from 0 to 10)")

overview = st.text_area("Overview", height=150)

col1, col2 = st.columns(2)
with col1:
    budget = st.number_input("Budget", min_value=0.0, value=10000000.0)
    revenue = st.number_input("Revenue", min_value=0.0, value=50000000.0)
    runtime = st.number_input("Runtime (minutes)", min_value=0.0, value=120.0)
    popularity = st.number_input("Popularity", min_value=0.0, value=10.0)
    vote_count = st.number_input("Vote Count", min_value=0, value=1000)

with col2:
    release_year = st.number_input("Release Year", min_value=1900, max_value=2030, value=2023)
    release_month = st.number_input("Release Month", min_value=1, max_value=12, value=6)
    num_cast = st.number_input("Number of Main Cast", min_value=0, value=10)

# هنا بس عدد الـ genres والـ keywords (مش الـ one-hot)
num_genres = st.number_input("Number of Genres (عدد الأنواع)", min_value=0, value=3)
num_keywords = st.number_input("Number of Keywords (عدد الكلمات المفتاحية)", min_value=0, value=10)

if st.button("🔮 Predict Audience Score"):
    if not overview.strip():
        st.error("Please enter the movie overview.")
    else:
        with st.spinner("Predicting..."):
            # Overview
            overview_cleaned = clean_text(overview)
            overview_length = len(overview_cleaned.split())
            X_overview = tfidf.transform([overview_cleaned])

            # Numerical features (نفس الترتيب اللي في النوت بوك)
            num_features = np.array([[budget, revenue, runtime, popularity, vote_count,
                                      release_year, release_month, num_genres,
                                      num_keywords, num_cast, overview_length]])
            num_scaled = scaler.transform(num_features)

            # دمج TF-IDF + Numerical
            X_input = hstack([X_overview, num_scaled])

            # التوقع
            prediction = model.predict(X_input)[0]

            st.success(f"**Predicted Audience Score: {prediction:.2f} / 10** 🎉")
            st.balloons()