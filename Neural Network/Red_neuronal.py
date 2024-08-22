import keras
import pandas as pd  
import numpy as np
from sklearn.naive_bayes import GaussianNB 
from sklearn import metrics 
from sklearn.utils import shuffle
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os

from keras.api.preprocessing import sequence
from keras import optimizers, losses
from keras.api.utils import to_categorical
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout, Activation, Embedding, SimpleRNN, BatchNormalization
from keras.api.models import model_from_json

full_path = os.path.realpath(__file__)
dir_path = os.path.dirname(full_path)

X_train = pd.read_csv(f'{dir_path}/UNSW-NB15-train-set.csv') 
y_train = pd.read_csv(f'{dir_path}/UNSW-NB15-train-set-label.csv') 
X_test = pd.read_csv(f'{dir_path}/UNSW-NB15-test-set.csv') 
y_test = pd.read_csv(f'{dir_path}/UNSW-NB15-test-set-label.csv')

# Convertir los datos a float32
X_train = np.asarray(X_train).astype("float32")
y_train = np.asarray(y_train).astype("float32") 
X_test = np.asarray(X_test).astype("float32")
y_test = np.asarray(y_test).astype("float32") 
    

#ENTRENAMIENTO MODELO
#Este modelo contiene capas densas. 
#En las capas densas cada neurona está conectada con todas las neuronas de la capa anterior
#Se usa relu como función de activación para las capas hidden e input
#La función de activación es softmax para la última capa.
model1 = Sequential()

# Capa de entrada (Input Layer)
#model1.add(Dense(128, input_shape=(42,), activation='relu'))
model1.add(keras.Input(shape=(42,)))
model1.add(BatchNormalization())
model1.add(Dropout(0.0001))

# Primera capa oculta
model1.add(Dense(256, activation='relu'))
model1.add(BatchNormalization())
model1.add(Dropout(0.0001))

# Segunda capa oculta
model1.add(Dense(128, activation='relu'))
model1.add(BatchNormalization())
model1.add(Dropout(0.0001))

# Tercera capa oculta
model1.add(Dense(64, activation='relu'))
model1.add(BatchNormalization())
model1.add(Dropout(0.0001))

# Capa de salida (Output Layer)
model1.add(Dense(2, activation='softmax'))  

model1.summary()


# Compilamos el modelo 
#Loss - La función de pérdida (loss function).
#Optimizazor- Para minimizar la función de pérdida.
#Metricas - Como se evaluan los resultados del entrenamiento (precisión por ejemplo).
model1.compile(optimizer= optimizers.SGD(learning_rate=0.1),
              loss='sparse_categorical_crossentropy',  
              metrics=['accuracy'])


#Encajar el modelo en los datos.
#X_train - Las columnas de características de los datos de entrenamiento
#y_train - Las etiquetas de los datos de entrenamiento
#validation_data - Los datos de validación (testing)
#epochs - Número de epochs para entrenar el modelo. Un epoch es una iteración sobre todos los datos x e y proporcionados.
history = model1.fit(X_train, y_train, 
          validation_data = (X_test, y_test),
          epochs = 20)


# Predicción sobre el conjunto de prueba
y_pred = model1.predict(X_test)

# Convertir las predicciones de probabilidad a clases (la clase con mayor probabilidad)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calcular la precisión (precision score)
precision = precision_score(y_test, y_pred_classes, average='weighted')

# Mostrar la precisión
print(f"Precision final: {precision:.4f}")


#Dibujar los gráficos de precisión
plt.figure(figsize = (15, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label = "Pérdida de entrenamiento")
plt.plot(history.history['val_loss'], label = "Pérdida de validación")
plt.title("Pérdida de entrenamiento vs Pérdida de validación")
plt.xlabel("EPOCH'S")
plt.ylabel("Pérdida de entrenamiento vs Pérdida de validación")
plt.legend(loc = "best")

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label = "Precisión de entrenamiento")
plt.plot(history.history['val_accuracy'], label = "Precisión de validación")
plt.title("Precisión de entrenamiento vs Precisión de validación")
plt.xlabel("EPOCH'S")
plt.ylabel("Precisión de entrenamiento vs Precisión de validación")
plt.legend(loc = "best")

plt.show()


###Mostrar informe de métricas###

# Predicción sobre el conjunto de prueba
y_pred = model1.predict(X_test)

# Convertir las predicciones de probabilidad a clases
y_pred_classes = np.argmax(y_pred, axis=1)

# Convertir y_test a entero si es necesario
y_test_classes = y_test.astype(int)

# Mostrar el reporte de clasificación
DTclf_name=['Resultados Red Neuronal:','RegLog']

print('Resultados: %s:'%DTclf_name)
print(classification_report(y_test_classes, y_pred_classes))

