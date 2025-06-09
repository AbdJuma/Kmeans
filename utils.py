# Utils.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
os.makedirs("plots", exist_ok=True)

def normalize_dataset(data):
    """
    Normalize features in the dataset using standard scaling (zero mean, unit variance).
    """
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data

def reduce_to_2d(data, method="pca"):
    """
    Reduce dimensionality of the data to 2D using PCA or t-SNE for visualization.
    """
    if data.shape[1] <= 2:
        return data

    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2)
    else:
        return data

    reduced_data = reducer.fit_transform(data)
    return reduced_data

def plot_clusters(data_2d, labels, title, centroids=None):
    """
    Plot 2D clusters with optional centroids.
    """
    plt.figure()
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap="tab10", s=30)
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/{title.replace(' ', '_').replace('-', '').lower()}.png")
    plt.close()

def plot_loss_curve(loss_values, title):
    """
    Plot the loss function curve over iterations.
    """
    plt.figure()
    plt.plot(range(1, len(loss_values) + 1), loss_values, marker="o")
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Sum of Squared Errors (Loss)")
    plt.grid(True)
    plt.savefig(f"plots/{title.replace(' ', '_').replace('-', '').lower()}.png")
    plt.close()
