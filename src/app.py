from utils import db_connect
engine = db_connect()

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors

# --- PASO 1: CARGA DE DATOS ---
url = "https://breathecode.herokuapp.com/asset/internal-link?id=2326&path=adult-census-income.csv"
folder_data = "data"
file_path = os.path.join(folder_data, "adult_census_income.csv")

if not os.path.exists(folder_data): os.makedirs(folder_data)
df_raw = pd.read_csv(url)
df_raw.to_csv(file_path, index=False)

# --- PASO 2: PREPROCESAMIENTO ---
df = df_raw.replace('?', np.nan).dropna().copy()

# Encoding variable objetivo
le = LabelEncoder()
df['income_encoded'] = le.fit_transform(df['income'])

# Definición de columnas (Nombres con puntos según el dataset)
categorical_cols = ['workclass', 'education', 'marital.status', 'occupation', 
                    'relationship', 'race', 'sex', 'native.country']
numerical_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 
                  'capital.loss', 'hours.per.week']

# Transformación
df_final = pd.get_dummies(df, columns=categorical_cols)
scaler = StandardScaler()
df_final[numerical_cols] = scaler.fit_transform(df_final[numerical_cols])

# --- PASO 3 Y 4: MODELO DE RECOMENDACIÓN ---
df_exitosos = df_final[df_final['income_encoded'] == 1].copy()
X_train = df_exitosos.drop(['income', 'income_encoded'], axis=1)

model_recomender = NearestNeighbors(n_neighbors=3, metric='cosine')
model_recomender.fit(X_train)

# --- PASO 5: GUARDADO DE MODELO ---
if not os.path.exists('models'): os.makedirs('models')
joblib.dump(model_recomender, 'models/recomender_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("✅ PROCESO COMPLETADO: Datos cargados, modelo entrenado y persistido en /models.")

# Función de prueba rápida
def recomendar_estrategia(indice):
    usuario_input = df_final.iloc[[indice]].drop(['income', 'income_encoded'], axis=1)
    distancias, indices = model_recomender.kneighbors(usuario_input)
    return df.iloc[df_exitosos.index[indices[0]]][['education', 'occupation', 'hours.per.week']]

print("\n--- EJEMPLO DE RECOMENDACIÓN PARA ÍNDICE 1114 ---")
display(recomendar_estrategia(1114))
