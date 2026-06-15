import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#load data
df = pd.read_csv("kmeans_social_media_users.csv")
# print(df.head())

#features for clustering
X = df[
    [
        'daily_active_minutes',
        'posts_per_week',
        'stories_per_week',
        'avg_likes_received',
        'avg_comments_received',
        'profile_completeness_percent',
        'friends_count',
        'groups_joined',
        'events_rsvp',
        'pages_liked',
        'ads_clicked_per_month',
        'account_age_days',
        'login_streak_days'
    ]
]
# print(f"X : {X}")

#scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# print(f"X_scaled : {X_scaled}")

#elbow method
wcss = []

for k in range(1,11):
    model = KMeans(
        n_clusters=k,
        random_state=42
    )
    model.fit(X_scaled)
    wcss.append(model.inertia_)

# print(f"wcss : {wcss}")

#plot elbow graph
plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

#final model
kmeans = KMeans(
    n_clusters=4,
    init='k-means++',
    random_state=42
)

clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

#print cluster assignments
print(df[['user_id','Cluster']].head())

#PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
# print(f"X_pca : {X_pca}")

#plot clusters
plt.figure(figsize=(10,7))
plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=clusters,
    cmap='viridis'
)
plt.title("K-Means Clusters of Social Media Users")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label='Cluster')
plt.show()

#cluster centers
centroids = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=X.columns
)
print("\nCluster Centers:\n")
print(centroids)

#cluster summary
cluster_summary = df.groupby('Cluster').mean(numeric_only=True)
print("\nCluster Summary:\n")
print(cluster_summary)
