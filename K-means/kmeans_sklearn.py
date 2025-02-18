import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Set the number of clusters (K value)
k = 4  # You can change this value as needed

# Generate random data
np.random.seed(0)
data = np.random.randn(300, 2)

# call sklearn function to implement KMeans
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(data)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

def plot_data(data, centroids, labels, iteration):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, alpha=0.5, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='+') 
    plt.title(f"K-Means Clustering")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.show()

# Visualize cluster results
plot_data(data, centroids, labels, iteration=kmeans.n_iter_)
