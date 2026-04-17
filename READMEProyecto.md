# 📈 Estrategias de Movilidad Económica: Sistema de Recomendación Basado en el Censo

Este proyecto aplica técnicas de **Ciencia de Datos** y **Machine Learning** para analizar perfiles socioeconómicos y sugerir trayectorias (educación, ocupación, carga horaria) que aumenten la probabilidad de superar el umbral de ingresos de **$50,000 USD anuales**.

## 🎯 Objetivos
* **Transformación Digital:** Aplicar automatización y analítica a datos sociales.
* **Modelado:** Construir un motor de recomendación de "trayectorias de vida".
* **Impacto Social:** Visualizar cómo variables como el género y la educación influyen en la brecha económica.

## 🛠️ Herramientas
* **Python 3.11** (Pandas, NumPy, Scikit-Learn)
* **Algoritmo:** k-Nearest Neighbors (k-NN) con Similitud del Coseno.
* **Persistencia:** Modelos exportados con `joblib` para despliegue rápido.

## 🧠 Metodología
El sistema utiliza un enfoque de **Filtrado Basado en Contenido**. Al recibir un perfil que gana $\le 50K$, el algoritmo busca en el grupo de "Perfiles Exitosos" ($>50K$) a los 3 individuos más similares matemáticamente. Al comparar las diferencias, el sistema genera una sugerencia personalizada.

## 📊 Hallazgos Principales
* **El Peso de la Gestión:** Para usuarios con alta formación académica pero ingresos bajos, el modelo sugiere consistentemente una transición a roles de **Dirección/Gerencia** (`Exec-managerial`).
* **Impacto Educativo:** En perfiles técnicos, el salto de `Assoc-acdm` a `Bachelors` es el factor más determinante para el cambio de categoría de ingresos.

