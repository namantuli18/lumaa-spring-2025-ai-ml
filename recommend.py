import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import re
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import sys

stop_words = list(stopwords.words('english'))

df = pd.read_csv('mpst-movie-plot-synopses-with-tags/mpst_full_data.csv')

user_input = sys.argv[1]

with open('data/tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

with open('data/vec.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def recommend_movies(user_input, top_n=5):
    user_input_cleaned = clean_text(user_input)
    user_vec = vectorizer.transform([user_input_cleaned])
    
    similarity_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    
    recommendations = list(zip(df.iloc[top_indices]['title'].values, similarity_scores[top_indices]))

    print("\nUser Input:", user_input)
    print("\nTop Movie Recommendations:\n")
    for i, (title, score) in enumerate(recommendations, start=1):
        print(f"{i}. {title} (Similarity: {score:.4f})")


recommend_movies(user_input)