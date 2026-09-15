# Clasificación de Subtipos Moleculares de Cáncer de Mama

Este repositorio contiene la implementación de un modelo de aprendizaje no supervisado para analizar e identificar patrones en perfiles transcripcionales de cáncer de mama, reduciendo la alta dimensionalidad de los datos genómicos.

## Objetivo del Proyecto
Analizar conjuntos de datos de expresión génica y proyectarlos en mapas topológicos 2D para agrupar muestras clínicas e identificar patrones correspondientes a los distintos subtipos moleculares biológicos, facilitando su análisis e interpretación.

## Conjunto de Datos (Dataset)
Los datos utilizados en este proyecto fueron obtenidos a través de **cBioPortal**, extrayendo matrices de expresión génica de la cohorte clínica **TCGA** (The Cancer Genome Atlas)


## Tecnologías y Librerías Utilizadas
*   **Lenguaje:** Python 3.x
*   **Manipulación de datos:** NumPy, Pandas
*   **Visualización:** Matplotlib
*   **Modelado:** minisom
*   **Entorno:** [Jupyter Notebook / Google Colab]

## Metodología
1. **Preprocesamiento:** Limpieza, normalización y manejo de la matriz de datos de alta dimensionalidad (miles de genes).
2. **Modelado:** Implementación del algoritmo de Mapas Autoorganizados (SOM) para el aprendizaje no supervisado.
3. **Evaluación Visual:** Generación de un mapa 2D (U-Matrix) para observar la topología y las agrupaciones de los datos.

## Resultados Principales
* Se logró proyectar datos genómicos de alta dimensionalidad en un mapa 2D de manera efectiva.
* El modelo permitió identificar visualmente patrones y agrupaciones claras que corresponden a los distintos subtipos moleculares del cáncer de mama, validando la utilidad de la reducción de dimensionalidad en bioinformática.

## Cómo ejecutar el proyecto
1. Clona este repositorio:
   ```bash
   git clone [https://github.com/spec421/clasificacion-cancer-mama.git](https://github.com/spec421/clasificacion-cancer-mama.git)
