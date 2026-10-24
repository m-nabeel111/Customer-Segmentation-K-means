import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# generating synthetic data for clustering 
print(" 1. Data Loading")
np.random.seed(42)
data_list = []
for i in range(1, 201):
    r = np.random.rand()
    if r < 0.2:
        inc = np.random.randint(15, 35)
        spend = np.random.randint(5, 30)
    elif r < 0.4:
        inc = np.random.randint(15, 35)
        spend = np.random.randint(65, 95)
    elif r < 0.6:
        inc = np.random.randint(40, 70)
        spend = np.random.randint(40, 60)
    elif r < 0.8:
        inc = np.random.randint(70, 100)
        spend = np.random.randint(10, 40)
    else:
        inc = np.random.randint(80, 137)
        spend = np.random.randint(60, 95)
    data_list.append([i, np.random.choice(['Male','Female']), np.random.randint(18,70), inc, spend])

data = pd.DataFrame(data_list, columns=['CustomerID','Gender','Age','Annual Income (k$)','Spending Score (1-100)'])
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# checking the data
print(data.head(10))
print("\nShape:", data.shape)
print("\nDescribe:\n", data.describe())

# 3. Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# 4. K=5 Training
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(X)

print("\n 4. Cluster Profile Table ")
profile = data.groupby('Cluster').agg({
    'Age':'mean',
    'Annual Income (k$)':'mean',
    'Spending Score (1-100)':'mean',
    'CustomerID':'count'
}).rename(columns={'CustomerID':'Count'})
print(profile)

#5. Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'])
axes[0].set_title('1. Income vs Spending')

axes[1].plot(range(1, 11), wcss, marker='o')
axes[1].set_title('2. Elbow Method')

for i in range(5):
    axes[2].scatter(data[data['Cluster']==i]['Annual Income (k$)'], 
                data[data['Cluster']==i]['Spending Score (1-100)'], s=50, label=f'C{i}')
axes[2].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black', marker='*')
axes[2].set_title('3. Customer Segments')
axes[2].legend()

plt.tight_layout()
plt.savefig(r"D:\coding\Python\result.png", dpi=150)
print("\n")

plt.show()