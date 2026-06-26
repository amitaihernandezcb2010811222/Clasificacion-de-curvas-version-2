import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
import joblib

print("Iniciando entrenamiento de la red neuronal...")

# Cargar datos extraídos, codigo (extractordedatos.py) 
df = pd.read_csv("datos_para_entrenar_red_neuronal_V.csv")

# Quitar filas con valores faltantes
original_count = len(df)
df = df.dropna(subset=['Clase'])

if df.empty:
    raise ValueError(
        f"No hay datos de entrenamiento con etiquetas. "
    )

# Variables de entrada y salida
X = df[['logP', 'A0', 'A1', 'A2', 'phi21', 'phi31']].values
y = df['Clase'].values

# Convertimos de texto a números 
encoder = LabelEncoder()
y_num = encoder.fit_transform(y)

#Extandarizamos los datos 
scaler = StandardScaler()
X_esc = scaler.fit_transform(X)

#Cross validation
K = 4 
kf = KFold(n_splits=K, shuffle=True, random_state=42)

resultados_accuracy = []
fold_no = 1

print(f"Iniciando Validación Cruzada con {K} iteraciones (folds)...")

for train_index, test_index in kf.split(X_esc):
    # Entrenamiento y validacion de los folds
    X_train, X_test = X_esc[train_index], X_esc[test_index]
    y_train, y_test = y_num[train_index], y_num[test_index]
    
    # Creamos un modelo nuevo y limpio para cada Fold
    modelo_cv = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(6,)), 
        tf.keras.layers.Dropout(0.2), 
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(np.unique(y_num)), activation='softmax')
    ])
    
    modelo_cv.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Entrenamos silenciosamente
    modelo_cv.fit(X_train, y_train, epochs=200, verbose=0)
    
    # Evaluamos si esta bien la parte de la prueba 
    loss, accuracy = modelo_cv.evaluate(X_test, y_test, verbose=0)
    resultados_accuracy.append(accuracy * 100)
    
    print(f"Fold {fold_no}: Precision = {accuracy * 100:.2f}%")
    fold_no += 1

# PROMEDIO REAL DE TU INTELIGENCIA ARTIFICIAL
precision_promedio = np.mean(resultados_accuracy)

print(f"Precision del modelo es:{precision_promedio:.2f}%")

# Creamos la nueva red neuronal
modelo = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(6,)), 
    tf.keras.layers.Dropout(0.2), 
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    
    
    tf.keras.layers.Dense(len(np.unique(y_num)), activation='softmax')
])

modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Early stopping (evita memorizar y utiliza las formulas fisicas)
parada_temprana = tf.keras.callbacks.EarlyStopping(
    monitor='loss', 
    patience=20, 
    restore_best_weights=True
)

#Entrenamiento
print("Estudiando patrones de Fourier...")
historial = modelo.fit(
    X_esc, y_num, 
    epochs=50, 
    callbacks=[parada_temprana], 
    verbose=0
)

# Guardamos el conocimiento 
modelo.save("cerebro_ia.keras")
joblib.dump(scaler, "escalador.pkl")
joblib.dump(encoder, "etiquetas.pkl")

print("La red neuronal ha sido entrenada y guardada.")