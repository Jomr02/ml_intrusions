import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


full_path = os.path.realpath(__file__)
dir_path = os.path.dirname(full_path)


trainset = pd.read_csv(f'{dir_path}/UNSW_NB15_training-set.csv')
testset = pd.read_csv(f'{dir_path}/UNSW_NB15_testing-set.csv')


types = trainset.dtypes


def ohe_new_features(df, features_name, encoder):
    new_feats = encoder.transform(df[features_name])
    new_cols = pd.DataFrame(new_feats, dtype=int)
    new_df = pd.concat([df, new_cols], axis=1)
    new_df.drop(features_name, axis=1, inplace=True)
    return new_df

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
cat_feats = ['proto', 'service', 'state']
encoder.fit(trainset[cat_feats])
trainset = ohe_new_features(trainset, cat_feats, encoder)
testset = ohe_new_features(testset, cat_feats, encoder)

trainset = trainset.to_numpy()
testset = testset.to_numpy()
scaler =  MinMaxScaler()
scaler.fit(trainset)
trainset = scaler.transform(trainset)
testset = scaler.transform(testset)


cov_mat = np.cov(trainset.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
tot = sum(eigen_vals)
# var_exp ratio is fraction of eigen_val to total sum
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
# calculate the cumulative sum of explained variances
cum_var_exp = np.cumsum(var_exp)



dim_reducer = PCA(n_components=3)
trainset_reduced = dim_reducer.fit_transform(trainset)
testset_reduced = dim_reducer.fit_transform(testset)


km = KMeans(n_clusters=10, init='k-means++', max_iter=300, n_init=10, random_state=42)


y_km = km.fit_predict(trainset_reduced)

# Calcular el Silhouette Score
silhouette_avg = silhouette_score(trainset_reduced, y_km)

# Mostrar el Silhouette Score
print(f" Puntuación Silhouette KMeans: {silhouette_avg}")

fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(trainset_reduced[y_km==0, 0], trainset_reduced[y_km==0, 1], trainset_reduced[y_km==0, 2],c='lightgreen', label='Cluster 1')
ax.scatter(trainset_reduced[y_km==1, 0], trainset_reduced[y_km==1, 1], trainset_reduced[y_km==1, 2],c='orange', label='Cluster 2')
ax.scatter(trainset_reduced[y_km==2, 0], trainset_reduced[y_km==2, 1], trainset_reduced[y_km==2, 2],c='lightblue', label='Cluster 3')
ax.scatter(trainset_reduced[y_km==3, 0], trainset_reduced[y_km==3, 1], trainset_reduced[y_km==3, 2],c='#2ca02c', label='Cluster 4')
ax.scatter(trainset_reduced[y_km==4, 0], trainset_reduced[y_km==4, 1], trainset_reduced[y_km==4, 2],c='#17becf', label='Cluster 5')
ax.scatter(trainset_reduced[y_km==5, 0], trainset_reduced[y_km==5, 1], trainset_reduced[y_km==5, 2],c='#7f7f7f', label='Cluster 6')
ax.scatter(trainset_reduced[y_km==6, 0], trainset_reduced[y_km==6, 1], trainset_reduced[y_km==6, 2],c='#8c564b', label='Cluster 7')
ax.scatter(trainset_reduced[y_km==7, 0], trainset_reduced[y_km==7, 1], trainset_reduced[y_km==7, 2],c='#bcbd22', label='Cluster 8')
ax.scatter(trainset_reduced[y_km==8, 0], trainset_reduced[y_km==8, 1], trainset_reduced[y_km==8, 2],c='#9467bd', label='Cluster 9')
ax.scatter(trainset_reduced[y_km==9, 0], trainset_reduced[y_km==9, 1], trainset_reduced[y_km==9, 2],c='red', label='Cluster 10')
ax.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], km.cluster_centers_[:,2], s=85, alpha=0.75, marker='o', c='black', label='Centroids')
ax.set_title("Kmeans", fontsize='large')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.legend()
plt.show()


dbs = DBSCAN(eps=0.06, min_samples=15)
y_dbs = dbs.fit_predict(trainset_reduced)


print(f'Número de centroides DBSCAN : {len(set(dbs.labels_))}') #len(set(y_dbs))

# Calcular el Silhouette Score
silhouette_avg = silhouette_score(trainset_reduced, y_dbs)

# Mostrar el Silhouette Score
print(f" Puntuación Silhouette KMeans: {silhouette_avg}")


fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(trainset_reduced[:,0], trainset_reduced[:,1], trainset_reduced[:,2], c=y_dbs, cmap='Paired')
ax.set_title("DBSCAN", fontsize='large')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()


