from pymongo import MongoClient
import numpy as np

# Connect to the MongoDB database
client = MongoClient('mongodb_connection_string')
db = client['your_database_name']
collection = db['your_collection_name']

# Fetch the document by title
document = collection.find_one({'title': "This World"})  # Assuming 'This World' is the title

# Extract the PCA MFCCs data
pca_mfccs = document['pca_mfccs']

# Convert to a NumPy array
pca_mfccs_array = np.array(pca_mfccs)

print(pca_mfccs_array)
print(pca_mfccs_array.shape)  # This should print (20, 10)