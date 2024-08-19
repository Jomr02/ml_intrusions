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

full_path = os.path.realpath(__file__)
dir_path = os.path.dirname(full_path)

# Cargar datos de entrenamiento
initial_data = pd.read_csv(f'{dir_path}/UNSW_NB15_training-set.csv')

# Quitar filas vacías
data_to_use = initial_data.dropna()

data_to_use.shape

# Cargar datos de test
test_data = pd.read_csv(f'{dir_path}/UNSW_NB15_testing-set.csv')

X_test = test_data.drop(axis=1, columns=['attack_cat']) 
X_test = X_test.drop(axis=1, columns=['label'])


y1_test = test_data['attack_cat'].values
y2_test = test_data['label'].values

X = data_to_use.drop(axis=1, columns=['attack_cat'])
X = X.drop(axis=1, columns=['label'])

y1 = data_to_use['attack_cat'].values
y2 = data_to_use['label'].values

X_train = X
y1_train = y1
y2_train = y2


# Determinar columnas con datos categóricos y numéricos
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train.select_dtypes(include=['object', 'bool']).columns


# Definir las transformaciones para las columnas
t = [('ohe', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols),
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
DTclf = DecisionTreeClassifier(criterion="entropy", max_depth=8)
DTclf = DTclf.fit(X_train_transform, y2_train_transform)

#Predecir usando los datos de test
y_pred = DTclf.predict(X_test_transform)


# Medir la precisión del árbol de decisión
print("Precisión",metrics.accuracy_score(y2_test_transform, y_pred))

report=metrics.classification_report(y2_test_transform,y_pred)

DTclf_name=['Árbol de decisión','RegLog']

print('Resultados: %s:'%DTclf_name)

print(report)




