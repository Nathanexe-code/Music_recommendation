import os
import librosa
import numpy as np
import pymongo
from pymongo import MongoClient
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Create a Spark session with MongoDB Spark Connector
spark = SparkSession.builder \
    .appName("Music Data Loading") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/test_database.processed_metadata_features") \
    .config("spark.mongodb.output.uri", "mongodb://localhost:27017/test_database.processed_metadata_features") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .getOrCreate()


# Load data from MongoDB
df = spark.read.format("mongo").load()

# Show the DataFrame schema and some entries to verify the load
df.printSchema()

mfccs = df.select("pca_mfccs").collect()

# For the mean of MFCCS
processed_data = []

for mfcc in mfccs:
    for i in mfcc:
     for j in i:
        processed_data.append(np.mean(j))

processed_data = np.array(processed_data)
processed_data.shape

data = np.reshape(processed_data, (106399, 20))
data.shape

pca = PCA(n_components=5)
data = pca.fit_transform(data)

data.shape


# Calculate sum of squared distances for different numbers of clusters
inertias = []
ks = range(1, 15)  # Testing 1 to 10 clusters
for k in ks:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

# Plotting the elbow curve
plt.figure(figsize=(10, 8))
plt.plot(ks, inertias, 'bo-')
plt.xlabel('Number of Clusters, k')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method For Optimal k')
plt.xticks(ks)
plt.grid(True)
plt.show()


# Number of clusters
k = int(input("Enter value of K would you like: "))  # Adjust based on your analysis

# Initialize the KMeans algorithm
kmeans = KMeans(n_clusters=k, random_state=0)

# Fit the model to the data
kmeans.fit(data)

# Predict the cluster labels
labels = kmeans.labels_

# Cluster centers
centers = kmeans.cluster_centers_

# Reduce data to 2D using PCA for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)
reduced_centers = pca.transform(centers)

# Plotting
plt.figure(figsize=(10, 8))
scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolors='w')
plt.scatter(reduced_centers[:, 0], reduced_centers[:, 1], c='red', s=200, marker='X')  # Mark the cluster centers
plt.title('2D PCA of Music Samples Clustered by K-Means')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter, label='Cluster Label')
plt.show()

print("clusters: ", kmeans.cluster_centers_)