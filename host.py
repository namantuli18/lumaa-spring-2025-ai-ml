import streamlit as st
import pickle
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk

# Load stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset
df = pd.read_csv('mpst-movie-plot-synopses-with-tags/mpst_full_data.csv')

# Load precomputed TF-IDF matrix and vectorizer
with open('data/tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

with open('data/vec.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Function to clean user input text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Function to recommend movies based on user input
def recommend_movies(user_input, top_n=5):
    user_input_cleaned = clean_text(user_input)
    user_vec = vectorizer.transform([user_input_cleaned])
    
    similarity_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    
    recommendations = list(zip(df.iloc[top_indices]['title'].values, similarity_scores[top_indices]))
    
    return recommendations

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.write("Enter a short description of a movie, and we’ll find the best matches for you!")

user_input = st.text_area("Describe a movie plot:", placeholder="Example: A detective solving a mysterious murder in a small town")

if st.button("Get Recommendations"):
    if user_input.strip():
        recommendations = recommend_movies(user_input)
        st.subheader("Top Movie Recommendations:")
        for i, (title, score) in enumerate(recommendations, start=1):
            st.write(f"**{i}. {title}** (Similarity: `{score:.4f}`)")
    else:
        st.warning("Please enter a description before submitting.")
