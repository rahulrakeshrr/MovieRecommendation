from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle

# Load the movie metadata
movies_metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Load the request data
    data = request.get_json(force=True)
    movie_name = data['movie_name']
    num_recommendations = int(data['num_recommendations'])

    # Find the row index of the movie
    idx = movies_metadata[movies_metadata['title'] == movie_name].index[0]

    # Get the similarity scores of the movie with other movies
    sim_scores = list(enumerate(movie_similarity[idx]))

    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the movie indices of the top recommended movies
    movie_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]

    # Get the movie titles and their similarity scores
    movie_titles = movies_metadata['title'].iloc[movie_indices]
    movie_scores = [i[1] for i in sim_scores[1:num_recommendations+1]]

    # Create the response data
    response_data = {}
    response_data['movies'] = []
    for i in range(num_recommendations):
        movie_data = {}
        movie_data['title'] = movie_titles.iloc[i]
        movie_data['score'] = movie_scores[i]
        response_data['movies'].append(movie_data)

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')



# Load the movie similarity matrix
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
