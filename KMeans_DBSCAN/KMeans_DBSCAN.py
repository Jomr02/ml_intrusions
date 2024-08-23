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

def ohe_new_features(df, features_name, encoder):
    # Transformar las características categóricas en variables binarias usando el encoder
    new_feats = encoder.transform(df[features_name])
    
    # Crear un nuevo DataFrame con las características transformadas
    new_cols = pd.DataFrame(new_feats, dtype=int)
    
    # Concatenar las nuevas características al DataFrame original
    new_df = pd.concat([df, new_cols], axis=1)
    
    # Eliminar las características originales categóricas del DataFrame
    new_df.drop(features_name, axis=1, inplace=True)
    
    # Devolver el DataFrame actualizado
    return new_df

def KMeans_DBSCAN():
    full_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(full_path)


    # Cargar los conjuntos de datos de entrenamiento y prueba desde archivos CSV ubicados en el mismo directorio que el script
    trainset = pd.read_csv(f'{dir_path}/UNSW_NB15_training-set.csv')
    testset = pd.read_csv(f'{dir_path}/UNSW_NB15_testing-set.csv')

    # Obtener los tipos de datos de cada columna en el conjunto de datos de entrenamiento
    types = trainset.dtypes

    # Definir una función para aplicar One-Hot Encoding a las características categóricas del DataFrame

    # Crear un objeto OneHotEncoder para convertir las características categóricas en binarias
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Especificar las características categóricas a transformar
    cat_feats = ['proto', 'service', 'state']

    # Ajustar el encoder a las características categóricas del conjunto de datos de entrenamiento
    encoder.fit(trainset[cat_feats])

    # Aplicar la transformación One-Hot Encoding al conjunto de datos de entrenamiento y de prueba
    trainset = ohe_new_features(trainset, cat_feats, encoder)
    testset = ohe_new_features(testset, cat_feats, encoder)

    # Convertir los DataFrames de entrenamiento y prueba a arrays de NumPy
    trainset = trainset.to_numpy()
    testset = testset.to_numpy()

    # Crear un objeto MinMaxScaler para escalar las características entre 0 y 1
    scaler = MinMaxScaler()

    # Ajustar el scaler a los datos de entrenamiento
    scaler.fit(trainset)

    # Escalar los datos de entrenamiento y de prueba utilizando el scaler ajustado
    trainset = scaler.transform(trainset)
    testset = scaler.transform(testset)



    # Crear un objeto PCA para reducir la dimensionalidad a 3 componentes principales
    dim_reducer = PCA(n_components=3)

    # Aplicar PCA al conjunto de datos de entrenamiento para reducir su dimensionalidad
    trainset_reduced = dim_reducer.fit_transform(trainset)

    # Aplicar PCA al conjunto de datos de prueba para reducir su dimensionalidad (utilizando la misma transformación)
    testset_reduced = dim_reducer.transform(testset)

    # Crear un objeto KMeans para realizar la agrupación en 10 clusters
    km = KMeans(n_clusters=9, init='k-means++', max_iter=300, n_init=10, random_state=42)

    # Ajustar el modelo KMeans a los datos reducidos de entrenamiento y predecir las etiquetas de los clusters
    y_km = km.fit_predict(trainset_reduced)

    # Calcular el Silhouette Score
    silhouette_avg = silhouette_score(trainset_reduced, y_km)


    ### MÉTODO DE OPTIMIZACIÓN DEL CODO DE KMEANS ###
    # silhouette_scores = []
    # for k in range(2, 11):
    #     km = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    #     y_km = km.fit_predict(trainset_reduced)
    #     silhouette_scores.append(silhouette_score(trainset_reduced, y_km))

    # plt.plot(range(2, 11), silhouette_scores)
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Silhouette Score')
    # plt.show()


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
    #ax.scatter(trainset_reduced[y_km==9, 0], trainset_reduced[y_km==9, 1], trainset_reduced[y_km==9, 2],c='red', label='Cluster 10')
    ax.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], km.cluster_centers_[:,2], s=85, alpha=0.75, marker='o', c='black', label='Centroids')
    ax.set_title("Kmeans", fontsize='large')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend()
    plt.show()

    ### METODO OPTIMIZACIÓN DBSCAN ###
    # # Definir rangos de búsqueda para eps y min_samples
    # eps_values = np.arange(0.01, 0.2, 0.01)
    # min_samples_values = range(5, 20)

    # best_silhouette = -1
    # best_params = {}

    # for eps in eps_values:
    #     for min_samples in min_samples_values:
    #         dbs = DBSCAN(eps=eps, min_samples=min_samples)
    #         y_dbs = dbs.fit_predict(trainset_reduced)
            
    #         # Verificar si hay más de 1 cluster
    #         if len(set(y_dbs)) > 1:
    #             silhouette_avg = silhouette_score(trainset_reduced, y_dbs)
                
    #             if silhouette_avg > best_silhouette:
    #                 best_silhouette = silhouette_avg
    #                 best_params = {'eps': eps, 'min_samples': min_samples}

    # print(f"Mejores parámetros: {best_params}, Silhouette Score: {best_silhouette}")

    dbs = DBSCAN(eps=0.12, min_samples=13)
    y_dbs = dbs.fit_predict(trainset_reduced)


    print(f'Número de centroides DBSCAN : {len(set(dbs.labels_))}') #len(set(y_dbs))

    # Calcular el Silhouette Score
    silhouette_avg = silhouette_score(trainset_reduced, y_dbs)

    # Mostrar el Silhouette Score
    print(f" Puntuación Silhouette DBSCAN: {silhouette_avg}")


    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(trainset_reduced[:,0], trainset_reduced[:,1], trainset_reduced[:,2], c=y_dbs, cmap='Paired')
    ax.set_title("DBSCAN", fontsize='large')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.show()


