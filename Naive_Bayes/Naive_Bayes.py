import pandas as pd  
import numpy as np
import os
from sklearn.naive_bayes import GaussianNB 
from sklearn import metrics 

def Naive_Bayes():
    full_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(full_path)

    # Leer datos de entrenamiento y test
    X_train = pd.read_csv(f'{dir_path}/UNSW-NB15-train-set.csv') 
    y_train = pd.read_csv(f'{dir_path}/UNSW-NB15-train-set-label.csv') 
    X_test = pd.read_csv(f'{dir_path}/UNSW-NB15-test-set.csv') 
    y_test = pd.read_csv(f'{dir_path}/UNSW-NB15-test-set-label.csv')
        

    # Para dejar el array como una sola dimensión
    y_train = y_train['label'] 
        
    #Crear un clasificador Naive Bayes
    model = GaussianNB()

    #Entrenar el modelo con los datos de test
    model.fit(X_train,y_train) 

    y_pred = model.predict(X_test)  


    # Medir la precisión de naive Bayes
    print("Precisión:",metrics.accuracy_score(y_test, y_pred))

    report=metrics.classification_report(y_test, y_pred)

    DTclf_name=['Naive Bayes','RegLog']

    print('Resultados: %s:'%DTclf_name)

    print(report)