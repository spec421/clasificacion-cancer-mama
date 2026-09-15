# -*- coding: utf-8 -*-

from minisom import MiniSom # Librería que implementa el algoritmo SOM
import pandas as pd # Pandas para procesar datos
import numpy as np # NumPy para procesar datos
from sklearn.preprocessing import StandardScaler # Sci-kit para preprocesar los datos
import matplotlib.pyplot as plt # Matplotlib para graficar

print("--- INICIANDO EL PRE-PROCESAMIENTO DE DATOS ---")

# ==========================================
# PASO 1: CARGAR LOS ARCHIVOS (Simulación)
# ==========================================

# 1A. Cargamos la Matriz de Expresión (Genes en filas, Pacientes en columnas)
df_genes = pd.read_csv('data_mrna_seq_v2_rsem.txt', sep='\t', index_col='Hugo_Symbol')
# Eliminamos columnas basura que a veces trae cBioPortal como 'Entrez_Gene_Id'
if 'Entrez_Gene_Id' in df_genes.columns:
    df_genes = df_genes.drop(columns=['Entrez_Gene_Id'])

# 1B. Cargamos la Tabla Clínica (Pacientes en filas, Metadatos en columnas)
# Omitimos las primeras 4 filas porque en cBioPortal suelen ser comentarios de texto (empiezan con #)
df_clinico = pd.read_csv('data_clinical_patient.txt', sep='\t', skiprows=4, index_col='PATIENT_ID')

# ===============================================
# PASO 2: TRANSPONER Y LIMPIAR LA MATRIZ GENÉTICA
# ===============================================

print("\n1. Transponiendo la matriz genética...")
# Usamos .T para que los Pacientes pasen a ser las FILAS (igual que en la tabla clínica)
df_genes_T = df_genes.T

# Los IDs genéticos a veces tienen un sufijo (ej. TCGA-A2-A0T2-01).
# El '-01' indica que es el tumor primario. La tabla clínica suele tener solo 'TCGA-A2-A0T2'.
# Cortamos el string del ID a los primeros 12 caracteres para que coincidan perfectamente.
df_genes_T.index = df_genes_T.index.str[:12]

# ==========================================
# PASO 3: EL CRUCE MATEMÁTICO (INNER JOIN)
# ==========================================
# Queremos pegar la columna 'Subtype' (el PAM50) al final de la matriz de genes.

print("2. Extrayendo la etiqueta (Ground Truth) de la tabla clínica...")
# Seleccionamos solo la columna que nos importa para no saturar la memoria
etiquetas = df_clinico[['SUBTYPE']]

print("3. Realizando la intersección (Merge)...")
# Realizamos el Inner Join usando los índices (los IDs de los pacientes)
# Solo sobrevivirán los pacientes que tengan datos genéticos Y diagnóstico clínico.
dataset_final = pd.merge(df_genes_T, etiquetas, left_index=True, right_index=True, how='inner')


# ==========================================
# PASO 4: LIMPIEZA FINAL DE DATOS NULOS
# ==========================================

print("4. Eliminando pacientes sin diagnóstico...")
# Si un médico no anotó el subtipo, no nos sirve para evaluar el SOM. Eliminamos esos NaNs.
dataset_final = dataset_final.dropna(subset=['SUBTYPE'])

print("\n--- ¡PROCESO COMPLETADO! ---")
print(f"Dimensión final del Dataset: {dataset_final.shape[0]} pacientes y {dataset_final.shape[1]-1} genes.")

# ==========================================
# PASO 5. REDUCCIÓN DE DIMENSIONALIDAD Y ESCALADO
# ==========================================

print("\n--- INICIANDO REDUCCIÓN DE DIMENSIONALIDAD Y ESCALADO ---")

# Separamos las características (X) de las etiquetas clínicas (Y)
# X: Solo los números (genes)
# Y: Solo los diagnósticos (SUBTYPE)
X_crudo = dataset_final.drop(columns=['SUBTYPE'])
Y_etiquetas = dataset_final['SUBTYPE']

print(f"Dimensión inicial: {X_crudo.shape[0]} pacientes y {X_crudo.shape[1]} genes.")

# =================================================
# PASO 6. FILTRADO POR VARIANZA (Feature Selection)
# =================================================

print("\n1. Calculando la varianza de cada gen...")
# Calculamos la varianza a lo largo de las columnas (axis=0)
varianzas = X_crudo.var(axis=0)

# Definimos cuántos genes queremos conservar (en este caso, los 1,500 más informativos)
N_GENES_TOP = 1500

print(f"2. Seleccionando los {N_GENES_TOP} genes con mayor varianza biológica...")
# .nlargest ordena de mayor a menor varianza y nos da los nombres de los genes
genes_seleccionados = varianzas.nlargest(N_GENES_TOP).index

# Filtramos la matriz original para quedarnos solo con esos genes
X_filtrado = X_crudo[genes_seleccionados]


# ==========================================
# PASO 7. ESTANDARIZACIÓN DE LOS DATOS
# ==========================================
print("3. Estandarizando los datos (Z-Score)...")
scaler = StandardScaler()

# fit_transform calcula la media y varianza, y aplica la fórmula de Z-score
X_escalado_numpy = scaler.fit_transform(X_filtrado)

# scikit-learn devuelve un array de numpy puro.
# Lo volvemos a convertir en un DataFrame de Pandas para no perder los nombres.
X_final = pd.DataFrame(X_escalado_numpy,
                       index=X_filtrado.index,
                       columns=X_filtrado.columns)

print("\n--- ¡PROCESO COMPLETADO! ---")
print(f"Dimensión final para el SOM: {X_final.shape[0]} pacientes y {X_final.shape[1]} genes.")

# ==========================================
# PASO 8. CONFIGURACIÓN DEL MAPA AUTO-ORGANIZATIVO (SOM)
# ==========================================

print("\n--- CONFIGURANDO EL MAPA AUTO-ORGANIZATIVO (SOM) ---")

# 1. Transformar el DataFrame a una matriz pura de NumPy (necesario para MiniSom)
data = X_final.values

# 2. Definir la topología (El tamaño de la cuadrícula)
# Una regla heurística matemática para el número de neuronas es: 5 * sqrt(N_muestras)
# Para ~1000 pacientes: 5 * sqrt(1000) ≈ 158 neuronas.
# Una cuadrícula de 15x15 = 225 neuronas es un tamaño excelente.
map_x = 15
map_y = 15

print(f"Topología de la red: {map_x} x {map_y} neuronas (Total: {map_x * map_y})")

# 3. Inicializar el SOM
som = MiniSom(x=map_x,
              y=map_y,
              input_len=data.shape[1], # Los 1500 genes
              sigma=1.5,               # Radio inicial de la vecindad (Gaussiana)
              learning_rate=0.5,       # Tasa de aprendizaje inicial (Alpha)
              neighborhood_function='gaussian',
              topology='rectangular',
              random_seed=42)

# 4. Inicialización de Pesos
print("Inicializando los pesos de las neuronas (PCA Initialization)...")
# Usar PCA (Análisis de Componentes Principales) para inicializar los pesos es mucho mejor
# que hacerlo al azar. Ayuda a que el mapa converja más rápido y preserve la topología global.
som.pca_weights_init(data)

# 5. El Entrenamiento (Ciclo de Competencia y Adaptación)
print("Entrenando el modelo (10,000 iteraciones)...")
# train_batch actualiza los pesos de forma más estable que train_random
iteraciones = 10000
som.train_batch(data, num_iteration=iteraciones, verbose=True)

print("\n--- ¡ENTRENAMIENTO COMPLETADO! ---")

# ==========================================
# PASO 9. MÉTRICA DE VALIDACIÓN
# ==========================================

# Error de Cuantificación (Quantization Error):
# Mide la distancia promedio entre cada paciente y su neurona ganadora (BMU).
# Entre más cercano a 0, más precisos son los "arquetipos".
qe = som.quantization_error(data)

print(f"Error de Cuantificación (QE): {qe:.4f}")

# ==========================================
# PASO 10. GENERACIÓN DE LA U-MATRIX
# ==========================================

print("\n--- GENERANDO EL MAPA TOPOLÓGICO FINAL ---")

# 1. Configuración del lienzo (Tamaño grande)
plt.figure(figsize=(12, 10))

# distance_map() devuelve la matriz de distancias euclidianas entre neuronas vecinas.
# Usamos .T (transpuesta) para alinear correctamente los ejes X e Y visuales.
u_matrix = som.distance_map().T

# pcolor dibuja la cuadrícula. Usamos el mapa de colores 'bone_r' (hueso invertido)
# donde Blanco/Gris claro = Clústeres (Valles) y Negro/Oscuro = Fronteras (Abismos).
plt.pcolor(u_matrix, cmap='bone_r', alpha=0.9)
plt.colorbar(label='Distancia Topológica (U-Matrix)', orientation='horizontal')

# Extraemos los subtipos únicos que existen en tus etiquetas clínicas (ej. Luminal A, Basal, etc.)
subtipos_unicos = Y_etiquetas.unique()

# Definimos una paleta de colores y marcadores geométricos para distinguirlos bien
colores = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4']
marcadores = ['o', 's', 'D', '^', 'v', 'p']

# Creamos diccionarios para asignar un color y un marcador a cada diagnóstico
color_dict = {subtipos_unicos[i]: colores[i % len(colores)] for i in range(len(subtipos_unicos))}
marker_dict = {subtipos_unicos[i]: marcadores[i % len(marcadores)] for i in range(len(subtipos_unicos))}

print("Mapeando los pacientes en la cuadrícula neuronal...")

# Recorremos cada paciente (x) y su diagnóstico real (etiqueta)
for i, x in enumerate(data):
    # Encontramos la neurona ganadora (BMU - Best Matching Unit) para este paciente
    w = som.winner(x)
    etiqueta = Y_etiquetas.iloc[i]

    # Dibujamos el marcador del paciente en las coordenadas de su neurona
    # Sumamos +0.5 para que el punto caiga exactamente en el centro del cuadrado
    plt.plot(w[0] + 0.5, w[1] + 0.5,
             marker_dict[etiqueta],
             markerfacecolor='None',          # Puntos huecos para que se vea el fondo
             markeredgecolor=color_dict[etiqueta],
             markersize=10,
             markeredgewidth=2)

# Creamos la leyenda personalizada para la esquina del mapa
leyendas = []
for subtipo in subtipos_unicos:
    leyendas.append(plt.Line2D([0], [0], marker=marker_dict[subtipo], color='w',
                               markeredgecolor=color_dict[subtipo], markerfacecolor='None',
                               markersize=10, markeredgewidth=2, label=subtipo))

plt.legend(handles=leyendas, loc='upper right', bbox_to_anchor=(1.25, 1), title="Subtipo Molecular (PAM50)")

plt.title('Proyección Topológica de Subtipos de Cáncer de Mama (SOM U-Matrix)', fontsize=16, fontweight='bold')
plt.xlabel('Neuronas (Dimensión X)')
plt.ylabel('Neuronas (Dimensión Y)')

plt.tight_layout()
plt.show()

print("\n--- ¡GRÁFICA GENERADA CON ÉXITO! ---")