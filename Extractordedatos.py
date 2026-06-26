import numpy as np
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.optimize import minimize_scalar, curve_fit

# Calcular la fase 
def calcular_fase(t, P):
    return ((t - np.min(t)) / P) % 1.0

#Calcular la longitud para elegir el mejor periodo
def longitud_cuerda(P, t, mag):
    fase = calcular_fase(t, P)
    idx = np.argsort(fase)
    mag_norm = (mag[idx] - np.min(mag)) / (np.max(mag) - np.min(mag))
    return np.sum(np.sqrt(np.diff(fase[idx])**2 + np.diff(mag_norm)**2))

# Descomposicion de Fourier 
def ecuacion_fourier(x, A0, A1, A2, A3, A4, Phi1, Phi2, Phi3, Phi4):
    fase_rad = 2 * np.pi * x
    return (A0 + 
            A1 * np.cos(1 * fase_rad + Phi1) + 
            A2 * np.cos(2 * fase_rad + Phi2) + 
            A3 * np.cos(3 * fase_rad + Phi3) + 
            A4 * np.cos(4 * fase_rad + Phi4))

def extraer_fourier(t, mag):
    # Periodo mas optimo (P)
    res = minimize_scalar(longitud_cuerda, bounds=(0.01, 1.1), args=(t, mag), method='bounded')
    P = res.x
    
    # Encontrar E = Tiempo de magnitud mínima (brillo MÁXIMO)
    E = t[np.argmin(mag)]
    
    # Fase con respecto a E
    fase = ((t - E) / P) % 1.0
    
    # Datos que acomodan la curva de luz para nuestra red neuronal
    A0_init = np.mean(mag)
    A1_init = (np.max(mag) - np.min(mag)) / 2.0
    p0 = [A0_init, A1_init, 0.1, 0.05, 0.01, 3.14, 3.14, 3.14, 3.14]
    
    try:
        # Ajustamos los parametros a la ecuacion de Fourier
        popt, _ = curve_fit(ecuacion_fourier, fase, mag, p0=p0, maxfev=10000)
        A0, A1, A2, A3, A4, Phi1, Phi2, Phi3, Phi4 = popt
        
        # Ya que la amplitud debe de ser positiva, si es dado caso que sea negativo, invertimos el signo y sumamos Phi
        if A1 < 0: A1 = -A1; Phi1 += np.pi
        if A2 < 0: A2 = -A2; Phi2 += np.pi
        if A3 < 0: A3 = -A3; Phi3 += np.pi
        
        # Calculamos los parámetros phi21 y phi31 que diferencian RRab de RRc
        phi21 = (Phi2 - 2 * Phi1) % (2 * np.pi)
        phi31 = (Phi3 - 3 * Phi1) % (2 * np.pi)
        
        # Se regresa el vector de 6 variables 
        return [np.log10(P), A0, A1, A2, phi21, phi31]
        
    except Exception as e:
        print(f"  [!] Error al ajustar Fourier: {e}")
        return [np.log10(P), A0_init, 0, 0, 0, 0]

# EEtiquetamos y extraemos los datos 
etiquetas_oficiales = {
    'V1': 'RRab', 'V2': 'RRab', 'V3': 'RRab', 'V4': 'RRab', 'V5': 'RRab',
    'V6': 'RRab', 'V7': 'AC', 'V8': 'RRab', 'V9': 'RRab', 'V10': 'RRc',
    'V11': 'RRc', 'V12': 'RRc', 'V13': 'RRc', 'V25': 'RRab', 'V29': 'RRab',
    'V30': 'BL Her', 'V31': 'RRc', 'V32': 'RRc', 'V33': 'SX Phe', 'V34': 'SX Phe',
    'V35': 'Sx Phe', 'V36': 'SX Phe', 'V37': 'SX Phe', 'V38': 'SX Phe', 'V39': 'RRc',
    'V40': 'RRd', 'V41': 'SX Phe', 'V42': 'SR'
}

if __name__ == "__main__":
    carpeta_datos = r"C:\Users\Usuario\Downloads\M92_Todas\Todas"  #Cambiar ruta de los archivos de datos 
    archivos = glob.glob(os.path.join(carpeta_datos, '*_V.dat'))
    datos = []

    print("Extraemos los datos y generamos el archivo 'datos_para_entrenar.csv'...")
    print("Iniciando descomposición de Fourier:")
    for archivo in archivos:
        nombre = os.path.basename(archivo).split('_')[0]
        
        # Leemos SOLO las primeras dos columnas
        df = pd.read_csv(archivo, delim_whitespace=True, usecols=[0, 1], names=['HJD', 'Mag'])
        features = extraer_fourier(df['HJD'].values, df['Mag'].values)
        
        # En caso de que no aparezca, la marcamos como estrella (desconocida)
        clase_real = etiquetas_oficiales.get(nombre, 'Desconocida')
        
        datos.append([nombre] + features + [clase_real])

    columnas = ['Estrella', 'logP', 'A0', 'A1', 'A2', 'phi21', 'phi31', 'Clase']
    df_final = pd.DataFrame(datos, columns=columnas)
    
    # Filtramos las desconocidas para que no tengamos ningun problema a la hora de entrenar nuestra red neuronal
    df_final = df_final[df_final['Clase'] != 'Desconocida']
    
    df_final.to_csv("datos_para_entrenar_red_neuronal_V.csv", index=False)
    print("Listo. Archivo 'datos_para_entrenar_red_neuronal_V.csv' generado con todas las clases incluidas.")