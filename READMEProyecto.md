# 📈 Estrategias de Movilidad Económica: Sistema de Recomendación Basado en el Censo

Este proyecto aplica técnicas de **Ciencia de Datos** y **Machine Learning** para analizar perfiles socioeconómicos y sugerir trayectorias (educación, ocupación, carga horaria) que aumenten la probabilidad de superar el umbral de ingresos de **$50,000 USD anuales**.

## 🎯 Objetivos
* **Transformación Digital:** Aplicar automatización y analítica a datos sociales.
* **Modelado:** Construir un motor de recomendación de "trayectorias de vida".
* **Impacto Social:** Visualizar cómo variables como el género y la educación influyen en la brecha económica.

## 🛠️ Herramientas

Python 3.11 (Pandas, NumPy, Scikit-Learn, Imbalanced-learn).
Algoritmos: * Baseline: k-Nearest Neighbors (k-NN).
Final: Random Forest Classifier optimizado con RandomizedSearchCV.
Tratamiento de Datos: Balanceo de clases mediante SMOTE para mejorar la detección de ingresos altos.
Persistencia: Modelos y escaladores exportados con joblib.

## 🧠 Metodología
El sistema evolucionó de un filtrado basado en contenido a un Modelo Predictivo Probabilístico.Balanceo: Se aplicó SMOTE para corregir el sesgo hacia la clase mayoritaria ($\le 50K$), elevando el Recall de la clase de altos ingresos al 81%.Optimización: Se maximizó el área bajo la curva (AUC-ROC a 0.91) para asegurar una clasificación robusta.Simulación: El sistema no solo busca similitudes, sino que utiliza el modelo para predecir el impacto de cambios específicos (ej. aumentar horas o nivel educativo) en la probabilidad de éxito del usuario.

## 📊 Hallazgos Principales
* **El Peso de la Gestión:** Para usuarios con alta formación académica pero ingresos bajos, el modelo sugiere consistentemente una transición a roles de **Dirección/Gerencia** (`Exec-managerial`).
* **Impacto Educativo:** En perfiles técnicos, el salto de `Assoc-acdm` a `Bachelors` es el factor más determinante para el cambio de categoría de ingresos.
Poder Predictivo: El nivel educativo (education.num) y las ganancias de capital son los factores con mayor peso en la importancia de variables del Random Forest.

Sensibilidad de Red: Gracias al balanceo sintético, el modelo ahora identifica perfiles con "potencial oculto" que el modelo base solía omitir.

![Importancia de Variables](figures/feature_importance.png)


