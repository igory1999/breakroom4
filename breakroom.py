import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

blob_centers = np.array(
    [[ 0.2,  2.3],
     [-1.5 ,  2.3],
     [-2.8,  1.8],
     [-2.8,  2.8],
     [-2.8,  1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

X, y = make_blobs(n_samples=2000, centers=blob_centers,
                  cluster_std=blob_std, random_state=7)

print(X.shape, y.shape)

k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X)

print("Labels:\n", kmeans.labels_)

print("Cluster's centers:\n", kmeans.cluster_centers_)

np.save("centroids.npy", kmeans.cluster_centers_)

X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
print("predict: ", kmeans.predict(X_new))




