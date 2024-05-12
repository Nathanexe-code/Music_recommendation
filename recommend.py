from flask import Flask, request, render_template
from pymongo import MongoClient
import numpy as np
import random

app = Flask(__name__)

# Connect to the MongoDB database
client = MongoClient('mongodb://localhost:27017/')
db = client['music_database']

@app.route('/')
def index():
    return render_template('index.html')  # A simple form in HTML to input the song title

@app.route('/process', methods=['POST'])
def process_song():
    title = request.form['title']
    collection = db['processed_metadata_features']

    # Fetch the document by title
    document = collection.find_one({"title": title})

    if document and 'pca_mfccs' in document:
        # Extract the PCA MFCCs data and process
        pca_mfccs = document['pca_mfccs']
        pca_mfccs_array = np.array(pca_mfccs)
        processed_mfcc = [np.mean(i) for i in pca_mfccs_array]
        data_list = np.array(processed_mfcc)
    else:
        return "Document not found or PCA MFCCs data missing"

    # Load cluster centroids and calculate distances
    centroids = np.loadtxt('new_cluster_centers.txt')
    distances = np.array([np.sqrt(np.sum((data_list - centroid) ** 2)) for centroid in centroids])
    closest_cluster_index = np.argmin(distances)

    # Fetch songs from the same cluster
    cluster_number = int(closest_cluster_index)
    collection = db['clustered_data']
    random_songs = get_random_songs_by_cluster(collection, cluster_number)

    # Prepare data for display
    songs_info = "<br>".join([f"Title: {song['title']} | Artist: {song['artist']} | Album: {song['album']}" for song in random_songs])
    return songs_info

def get_random_songs_by_cluster(collection, cluster_number, sample_size=5):
    documents = list(collection.find({"cluster": cluster_number}))
    return random.sample(documents, sample_size) if len(documents) > sample_size else documents

if __name__ == '__main__':
    app.run(debug=False)
