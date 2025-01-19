import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
dataset = pd.read_csv("week-3/Mall_Customers.csv")

# Basic info and checks
print(dataset.info())
print(dataset.describe())
print(dataset.isnull().sum())

# Standardization
features = dataset[["Annual Income (k$)", "Spending Score (1-100)", "Age"]]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Convert back to DataFrame for readability
scaled_df = pd.DataFrame(scaled_features, columns=["Annual Income", "Spending Score", "Age"])

# Visualize outliers with boxplots
plt.figure(figsize=(10, 6))
sns.boxplot(data=features)
plt.title("Boxplot of Features")
plt.show()

# Outlier removal using IQR
Q1 = features.quantile(0.25)
Q3 = features.quantile(0.75)
IQR = Q3 - Q1
threshold = 1.5
filtered_data = features[~((features < (Q1 - threshold * IQR)) | (features > (Q3 + threshold * IQR))).any(axis=1)]
print(f"Original shape: {features.shape}, After outlier removal: {filtered_data.shape}")

# Re-standardize after outlier removal
scaled_filtered = scaler.fit_transform(filtered_data)
scaled_filtered_df = pd.DataFrame(scaled_filtered, columns=["Annual Income", "Spending Score", "Age"])

# PCA for dimensionality reduction (optional, for better visualization)
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_filtered)

# Elbow method to find the optimal number of clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pca_features)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.show()

# Silhouette scores for additional validation
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pca_features)
    score = silhouette_score(pca_features, kmeans.labels_)
    print(f"For k={k}, Silhouette Score = {score:.2f}")

# Optimal number of clusters (e.g., based on elbow method or silhouette score)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(pca_features)

# Add cluster labels to the dataset
filtered_data["Cluster"] = kmeans.labels_

# Visualization of clusters in 2D
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=kmeans.labels_, palette="viridis", alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color="red", marker="x", s=200, label="Centroids")
plt.title("K-Means Clustering with PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()


# 3D scatter plot for detailed visualization (optional)
import plotly.express as px

fig = px.scatter_3d(
    filtered_data,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    z="Age",
    color="Cluster",
    title="3D Clustering Visualization",
    labels={"Cluster": "Cluster"},
)
fig.update_traces(marker=dict(size=5), selector=dict(mode="markers"))
fig.show()
