import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from scipy.optimize import minimize_scalar
from Extractordedatos import extraer_fourier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Calcular la fase
def calcular_fase(t, P):
    return ((t - np.min(t)) / P) % 1.0

def longitud_cuerda(P, t, mag):
    fase = calcular_fase(t, P)
    idx = np.argsort(fase)
    mag_norm = (mag[idx] - np.min(mag)) / (np.max(mag) - np.min(mag))
    return np.sum(np.sqrt(np.diff(fase[idx])**2 + np.diff(mag_norm)**2))

def procesar_estrella_desconocida(ruta_archivo):
    # Lee las columnas HJD y Magnitud 
    df = pd.read_csv(ruta_archivo, delim_whitespace=True, usecols=[0, 1], names=['HJD', 'Mag'])
    
    # Extraemos las 6 características de Fourier necesarias para el modelo
    features = np.array(extraer_fourier(df['HJD'].values, df['Mag'].values)).reshape(1, -1)
    
    # Utilizamos el modelo entrenado de la red neuronal
    modelo = tf.keras.models.load_model("cerebro_ia.keras")
    scaler = joblib.load("escalador.pkl")
    encoder = joblib.load("etiquetas.pkl")
    
    # Transformamos y predecimos
    features_esc = scaler.transform(features)
    prediccion = modelo.predict(features_esc, verbose=0)[0] # Tomamos la lista de probabilidades
    
    # Mostramos  el índice ganador 
    indice_ganador = np.argmax(prediccion)
    #Probabilidad de que la respuesta que nos meustre sea la correcta
    probabilidad = prediccion[indice_ganador] * 100
    
    # Decodificamos el número ganador al nombre de la estrella
    clase_ganadora = encoder.inverse_transform([indice_ganador])[0]
    
    return clase_ganadora, probabilidad

# El usuario ingresa el archivos con sus 2 columnas (HJD y Magnitud) 
if __name__ == "__main__":
    print("Clasificador de el tipo de estrella variable según sus curvas de luz (HJD y Magnitud).")
    
    archivo_usuario = input("\nIngresa el nombre o ruta del archivo .dat (con tu HJD y Magnitud): ")
    
    archivo_usuario = archivo_usuario.strip("\"'") 
    
    try:
        #Clasificacion final de la estrella nueva
        resultado, confianza = procesar_estrella_desconocida(archivo_usuario)
            
        #Resultado final (tipo de estrella + probailidad de que sea correcta)
        print(f"Tipo de estrella variable: {resultado} (Probabilidad de Confianza: {confianza:.2f}%)")
            
    except Exception as e:
            print(f"Ocurrio un error, intente de nuevo: {e}")



