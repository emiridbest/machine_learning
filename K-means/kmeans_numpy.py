import numpy as np
import matplotlib.pyplot as plt

# Set the number of clusters (K value)
k = 4
# Generate random data
np.random.seed(0)
data = np.random.randn(300,2)
# Initialize cluster centers
initial_indices = np.random.choice(len(data), k, replace=False)
centroids = data[initial_indices]

def plot_data(data, centroid, iteration, idx=None):
  plt.figure(figsize=(8, 6))
  if idx is not None:
    plt.scatter(data[:,0], data[:,1], c=idx, alpha=0.6, cmap='viridis')
  else:
    plt.scatter(data[:,0], data[:,1], alpha=0.6)
  plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='x')# Red 'X' marks the cluster centers
  plt.title(f"Iteration {iteration}")
  plt.xlabel("X Axis")
  plt.ylabel("Y Axis")
  plt.show()

# Visualize initial data and cluster centers
plot_data(data,centroids, iteration=0)

# Run K-means algorithm
for i in range(5): # Perform up to 5 iterations
  distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
  closest = np.argmin(distances, axis=1)
  plot_data(data, centroids, iteration=i+1, idx=closest)
  new_centroids = np.array([data[closest == j].mean(axis=0) for j in range(k)])
  if np.allclose(centroids, new_centroids):
    break
  centroids = new_centroids

plot_data(data, centroids, iteration=i+1, idx=closest)
