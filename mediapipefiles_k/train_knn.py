import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier

from dataset_builder import normalize_landmarks

def train_fast_knn():
    files = [
        "dataset_clap.csv", 
        "dataset_disco.csv", 
        "dataset_hello.csv", 
        "dataset_wakanda.csv", 
        "dataset_zombie.csv"
    ]
    labels = ["CLAP", "DISCO", "HELLO", "WAKANDA", "ZOMBIE"]
    
    features_list = []
    labels_list = []
    
    for idx, file_path in enumerate(files):
        print(f"Processing {file_path}...")
        df = pd.read_csv(file_path)
        df.fillna(0, inplace=True)
        # Extract the 132 landmark coordinates
        data = df.values[:, :132]
        # Normalize
        data = normalize_landmarks(data).astype(np.float32)
        features_list.append(data)
        # Replicate label for all rows in the dataset
        labels_list.extend([idx] * data.shape[0])
        
    X = np.vstack(features_list)
    y = np.array(labels_list)
    
    print(f"Training KNN on {X.shape[0]} samples...")
    # Used weighted distances and moderate neighbors for probabilistic 'Others' triggering
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X, y)
    
    # Save the model and labels array
    model_data = {
        "model": knn,
        "labels": labels
    }
    with open("fast_knn_classifier.pkl", "wb") as f:
        pickle.dump(model_data, f)
        
    print("✅ Successfully trained and saved KNN to fast_knn_classifier.pkl")

if __name__ == "__main__":
    train_fast_knn()
