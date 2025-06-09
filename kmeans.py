
# kmeans.py
import numpy as np

def initialize_centroids(data, num_clusters, seed=0):
    """
    Randomly select initial centroids from the dataset.
    """
    np.random.seed(seed)
    random_indices = np.random.choice(len(data), num_clusters, replace=False)
    centroids = data[random_indices]
    return centroids

def assign_points_to_clusters(data, centroids):
    """
    Assign each data point to the nearest centroid based on Euclidean distance.
    """
    cluster_labels = []
    for point in data:
        distances = [np.linalg.norm(point - center) for center in centroids]
        closest_cluster = np.argmin(distances)
        cluster_labels.append(closest_cluster)
    return np.array(cluster_labels)

def compute_new_centroids(data, labels, num_clusters):
    """
    Compute new centroids as the mean of points assigned to each cluster.
    """
    new_centroids = []
    for cluster_index in range(num_clusters):
        cluster_points = data[labels == cluster_index]
        if len(cluster_points) > 0:
            cluster_mean = np.mean(cluster_points, axis=0)
        else:
            cluster_mean = np.zeros(data.shape[1])
        new_centroids.append(cluster_mean)
    return np.array(new_centroids)

def calculate_total_loss(data, labels, centroids):
    """
    Compute the sum of squared distances (loss) between points and their cluster centroids.
    """
    total_loss = 0
    for i in range(len(data)):
        distance = np.linalg.norm(data[i] - centroids[labels[i]])
        total_loss += distance ** 2
    return total_loss

def run_kmeans(data, num_clusters, max_iterations=100, tolerance=1e-4, seed=0):
    """
    Run the k-means clustering algorithm.
    """
    centroids = initialize_centroids(data, num_clusters, seed)
    loss_per_iteration = []

    for iteration in range(max_iterations):
        cluster_labels = assign_points_to_clusters(data, centroids)
        updated_centroids = compute_new_centroids(data, cluster_labels, num_clusters)
        loss = calculate_total_loss(data, cluster_labels, updated_centroids)
        loss_per_iteration.append(loss)

        movement = np.linalg.norm(updated_centroids - centroids)
        if movement < tolerance:
            break

        centroids = updated_centroids

    return centroids, cluster_labels, loss_per_iteration
