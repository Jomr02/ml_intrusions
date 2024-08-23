import matplotlib
import pandas as pd
import numpy as np
import os
#import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, classification_report
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree


def DecisionTree():
    full_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(full_path)

    # Cargar datos de entrenamiento
    initial_data = pd.read_csv(f'{dir_path}/UNSW_NB15_training-set.csv')

    # Quitar filas vacías
    data_to_use = initial_data.dropna()

    data_to_use.shape

    # Cargar datos de test
    test_data = pd.read_csv(f'{dir_path}/UNSW_NB15_testing-set.csv')

    # Eliminar la columna 'attack_cat' y 'label' del conjunto de datos de prueba (X_test)
    X_test = test_data.drop(axis=1, columns=['attack_cat']) 
    X_test = X_test.drop(axis=1, columns=['label'])

    # Extraer los valores de la columna 'attack_cat' y 'label' del conjunto de datos de prueba y guardarlos en y1_test
    y1_test = test_data['attack_cat'].values
    y2_test = test_data['label'].values

    # Eliminar la columna 'attack_cat' y 'label' del conjunto de datos que se va a utilizar (X)
    X = data_to_use.drop(axis=1, columns=['attack_cat'])
    X = X.drop(axis=1, columns=['label'])

    # Extraer los valores de la columna 'attack_cat' y 'label' del conjunto de datos a utilizar y guardarlos en y1
    y1 = data_to_use['attack_cat'].values
    y2 = data_to_use['label'].values

    X_train = X
    y1_train = y1
    y2_train = y2


    # Determinar columnas con datos categóricos y numéricos
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_train.select_dtypes(include=['object', 'bool']).columns


    # Definir las transformaciones para las columnas
    t = [('ohe', OrdinalEncoder(handle_unknown='use_encoded_value',
        unknown_value=-1), categorical_cols),
        ('scale', StandardScaler(), numerical_cols)]

    col_trans = ColumnTransformer(transformers=t)

    # Encajar la transformación con los datos
    col_trans.fit(X_train)

    X_train_transform = col_trans.transform(X_train)

    # Encajar la transformación con los datos

    X_test_transform = col_trans.transform(X_test)

    X_train_transform.shape
    X_test_transform.shape


    pd.unique(y1)
    pd.unique(y2)

    # Definir el codificador de etiquetas
    target_trans = LabelEncoder()
    target_trans.fit(y1_train)

    # Aplicar la transformación en y1_train y y1_test
    y1_train_transform = target_trans.transform(y1_train)
    y1_test_transform = target_trans.transform(y1_test)


    # # Definir el codificador de etiquetas para el método de transformación en y2_train y y2_test
    target_trans = LabelEncoder()
    target_trans.fit(y2_train)
    y2_train_transform = target_trans.transform(y2_train)
    y2_test_transform = target_trans.transform(y2_test)


    ###DECISION TREE###

    # Entrenar al árbol de decisión
    DTclf = DecisionTreeClassifier(criterion="entropy", max_depth=10, 
                max_features= 'log2', min_samples_leaf= 2, min_samples_split= 10)
    DTclf = DTclf.fit(X_train_transform, y2_train_transform)

    #Predecir usando los datos de test
    y_pred = DTclf.predict(X_test_transform)


    # Medir la precisión del árbol de decisión
    print("Precisión",metrics.accuracy_score(y2_test_transform, y_pred))

    report=metrics.classification_report(y2_test_transform,y_pred)

    DTclf_name=['Árbol de decisión','RegLog']

    print('Resultados: %s:'%DTclf_name)

    print(report)


    # Crear el gráfico del árbol de decisión
    plt.figure(figsize=(170,85))  
    plot_tree(DTclf, filled=True, feature_names=X.columns, class_names=True, rounded=True)
    plt.suptitle("Árbol de decisión", fontsize=16)
    plt.show()


    # Definir el rango de parámetros a explorar
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2']
    }

# # Crear el objeto GridSearchCV
# grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), 
#                            param_grid=param_grid,
#                            cv=StratifiedKFold(n_splits=5),
#                            scoring='accuracy',  
#                            verbose=1,
#                            n_jobs=-1)

# # Entrenar GridSearchCV con los datos de entrenamiento
# grid_search.fit(X_train_transform, y2_train_transform)

# # Obtener los mejores parámetros
# best_params = grid_search.best_params_
# print("Mejores parámetros encontrados:")
# print(best_params)

# # Entrenar el modelo con los mejores parámetros encontrados
# best_model = grid_search.best_estimator_

# # Predecir usando los datos de test
# y_pred_best = best_model.predict(X_test_transform)

# # Medir la precisión del mejor modelo
# print("Precisión del mejor modelo:", metrics.accuracy_score(y2_test_transform, y_pred_best))

# # Mostrar el reporte de clasificación
# report_best = metrics.classification_report(y2_test_transform, y_pred_best)
# print(report_best)




