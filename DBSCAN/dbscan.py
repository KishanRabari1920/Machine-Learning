import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


#Create Dataset
X, y = make_moons(
    n_samples=500,
    noise=0.05,
    random_state=42
)
# print(f"features(X) : {X}")
# print(f"Target(y) : {y}")

#Visualize Dataset
plt.scatter(X[:,0], X[:,1])
plt.title("Original Moon Dataset")
plt.show()

#Scaling data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# print(f"X_scaled : {X_scaled}")

#Create eps Range
eps_range = np.arange(0.1, 1.1, 0.1)
print("eps values:")
print(eps_range)

#Apply DBSCAN for Different eps Values
for eps in eps_range:

    # Create DBSCAN Model
    dbscan = DBSCAN(
        eps=eps,
        min_samples=5
    )

    # Train Model and Predict Clusters
    clusters = dbscan.fit_predict(X_scaled)

    # Count Number of Clusters
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

    # Count Noise Points
    noise_points = list(clusters).count(-1)

    # Print Results
    print(f"eps value = {eps}")
    print("Number of Clusters:", n_clusters)
    print("Noise Points:", noise_points)
    print("First 20 Cluster Labels:")
    print(clusters[:20])

    # Plot Clusters
    plt.figure(figsize=(7,5))
    plt.scatter(
        X_scaled[:,0],
        X_scaled[:,1],
        c=clusters
    )
    plt.title(f"DBSCAN Clustering eps={eps:.1f}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


#Final Model
final_dbscan = DBSCAN(
    eps=0.2,
    min_samples=5
)
best_clusters = final_dbscan.fit_predict(X_scaled)
# print(f"best_clusters : {best_clusters}")


#Final Cluster Visualization
plt.figure(figsize=(8,6))
plt.scatter(
    X_scaled[:,0],
    X_scaled[:,1],
    c=best_clusters
)
plt.title("Final DBSCAN Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
