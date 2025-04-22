import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tabulate import tabulate

# Load data
file_path = 'anime.csv'
anime_data = pd.read_csv(file_path)

# Clean and fill data
anime_data['genre'] = anime_data['genre'].fillna('Unknown')
anime_data['type'] = anime_data['type'].fillna('Unknown')
anime_data['rating'] = anime_data['rating'].fillna(anime_data['rating'].median())
anime_data['members'] = anime_data['members'].fillna(anime_data['members'].median())

anime_data = anime_data.reset_index(drop=True)

# TF-IDF on genre
genre_vectorizer = TfidfVectorizer(stop_words='english')
genre_matrix = genre_vectorizer.fit_transform(anime_data['genre'])

# Fit model using only genre
model = NearestNeighbors(n_neighbors=30, metric='cosine')  # More neighbors to allow sorting
model.fit(genre_matrix)

def recommend_anime(anime_name, anime_data, model, genre_matrix, top_n=10):
    anime_name = anime_name.strip().lower()
    anime_data_copy = anime_data.copy()
    anime_data_copy['name'] = anime_data_copy['name'].str.strip().str.lower()

    if anime_name not in anime_data_copy['name'].values:
        print(f"Anime '{anime_name}' not found.") 
        return pd.DataFrame()
    
    anime_index = anime_data_copy[anime_data_copy['name'] == anime_name].index[0]
    distances, indices = model.kneighbors(genre_matrix[anime_index])

    neighbors = anime_data.iloc[indices[0][1:]]  
    sorted_neighbors = neighbors.sort_values(by=['rating', 'members'], ascending=[False, False])
    
    return sorted_neighbors[['name', 'genre', 'episodes', 'rating', 'members']].head(top_n)

def get_top_rated_anime(anime_data, top_n=10):
    return anime_data[['name', 'genre', 'rating', 'members']].head(top_n)

# Show top-rated anime (first 10 rows in the dataset)
print("Top Rated Anime:")
top_rated = get_top_rated_anime(anime_data)
print(tabulate(top_rated.reset_index(drop=True), headers="keys", tablefmt="grid"))


# Ask user to input Anime name
anime_name = input("\nEnter the name of the anime: ").strip()
recommendations = recommend_anime(anime_name, anime_data, model, genre_matrix)

if recommendations.empty:
    print("No recommendations available.")
else:
    print(f"Top recommendations for anime '{anime_name}':")
    print(tabulate(recommendations.reset_index(drop=True), headers="keys", tablefmt="grid"))
