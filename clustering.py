# kmeans_project/clustering.py
from sklearn.cluster import DBSCAN
from spherecluster import SphericalKMeans

def run_dbscan_clustering(data, eps=0.5, min_samples=5):
    """
    Run DBSCAN clustering algorithm using scikit-learn.
    """
    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan_model.fit_predict(data)
    return cluster_labels

def run_spherical_kmeans_clustering(data, num_clusters, max_iterations=100):
    """
    Run spherical k-means clustering algorithm using the spherecluster package.
    """
    spherical_model = SphericalKMeans(n_clusters=num_clusters, max_iter=max_iterations)
    cluster_labels = spherical_model.fit_predict(data)
    return cluster_labels
