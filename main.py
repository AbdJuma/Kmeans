# kmeans_project/main.py
import pandas as pd
from kmeans import run_kmeans
from clustering import run_dbscan_clustering, run_spherical_kmeans_clustering
from utils import normalize_dataset, reduce_to_2d, plot_clusters, plot_loss_curve

# Dictionary of datasets and their URLs
datasets_info = {
    "Iris": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    "Gaussians_0.6": "https://raw.githubusercontent.com/mohammedkhalilia/birzeit/master/comp9318/data/3gaussians-std0.6.csv",
    "Gaussians_0.9": "https://raw.githubusercontent.com/mohammedkhalilia/birzeit/master/comp9318/data/3gaussians-std0.9.csv",
    "Circles": "https://raw.githubusercontent.com/mohammedkhalilia/birzeit/master/comp9318/data/circles.csv",
    "Moons": "https://raw.githubusercontent.com/mohammedkhalilia/birzeit/master/comp9318/data/moons.csv"
}

# Iterate through all datasets
for dataset_name, dataset_url in datasets_info.items():
    print(f"\nProcessing dataset: {dataset_name}")

    # Load and preprocess the dataset
    if dataset_name == "Iris":
        raw_data = pd.read_csv(dataset_url, header=None)
        raw_data = raw_data.iloc[:, :-1]  # Remove label column
    else:
        raw_data = pd.read_csv(dataset_url)

    data = raw_data.values
    normalized_data = normalize_dataset(data)
    reduced_data = reduce_to_2d(normalized_data)

    # Run k-means clustering
    num_clusters = 3
    kmeans_centroids, kmeans_labels, kmeans_loss = run_kmeans(normalized_data, num_clusters)
    reduced_centroids = reduce_to_2d(kmeans_centroids)
    plot_clusters(reduced_data, kmeans_labels, f"{dataset_name} - k-Means", reduced_centroids)
    plot_loss_curve(kmeans_loss, f"{dataset_name} - k-Means Loss")

    # Run DBSCAN clustering
    dbscan_labels = run_dbscan_clustering(normalized_data)
    plot_clusters(reduced_data, dbscan_labels, f"{dataset_name} - DBSCAN")

    # Run Spherical k-means clustering
    spherical_labels = run_spherical_kmeans_clustering(normalized_data, num_clusters)
    plot_clusters(reduced_data, spherical_labels, f"{dataset_name} - Spherical k-Means")
