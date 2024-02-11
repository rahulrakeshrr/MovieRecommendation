import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# Load the movie ratings data
ratings_df = pd.read_csv("C:\Users\rahul\Documents\Complete projects\netflix_titles.csv")

# Pivot the data to create a user-item matrix
user_item_matrix = ratings_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

# Convert the user-item matrix to a sparse matrix
sparse_matrix = csr_matrix(user_item_matrix.values)

# Apply matrix factorization to the sparse matrix
model = TruncatedSVD(n_components=100)
model.fit(sparse_matrix)
U = model.transform(sparse_matrix)
V = model.components_

# Save the model to a pickle file
with open('model.pickle', 'wb') as f:
    pickle.dump(model, f)

# Get movie recommendations for a given movie
def get_recommendations(movie_name, num_recommendations=10):
    # Find the row index of the movie in the user-item matrix
    movie_index = user_item_matrix.columns.get_loc(movie_name)

    # Compute the cosine similarity between the movie and all other movies
    similarities = cosine_similarity(V.T[movie_index].reshape(1, -1), V.T).flatten()

    # Get the indices of the most similar movies
    most_similar_indices = similarities.argsort()[::-1][1:num_recommendations+1]

    # Get the titles and years of the most similar movies
    titles = [user_item_matrix.columns[i] for i in most_similar_indices]
    years = [ratings_df[ratings_df['movie_id'] == title]['year'].iloc[0] for title in titles]

    # Create a list of recommended movies
    recommendations = []
    for i in range(num_recommendations):
        movie = {'title': titles[i], 'year': years[i]}
        recommendations.append(movie)

    return recommendations
