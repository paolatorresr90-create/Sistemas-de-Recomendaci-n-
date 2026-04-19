from utils import db_connect
engine = db_connect()

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Configuración y Carga de Archivos
st.set_page_config(page_title="Estrategia de Movilidad Económica", page_icon="📊")

@st.cache_resource
def load_resources():
    model = joblib.load('models/final_rf_model_smote.pkl')
    scaler = joblib.load('models/final_scaler.pkl')
    columns = joblib.load('models/model_columns.pkl')
    return model, scaler, columns

try:
    rf_model, scaler, model_columns = load_resources()
except Exception as e:
    st.error("Error cargando los modelos. Asegúrate de que la carpeta /models tenga los archivos .pkl")
    st.stop()

st.title("📈 Recomendador de Ingresos con IA")
st.markdown("---")

# 2. Interfaz de Usuario
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Edad", 18, 90, 30)
    edu_num = st.slider("Nivel Educativo (Años)", 1, 16, 9)
    hours = st.number_input("Horas de trabajo semanales", 1, 99, 40)

with col2:
    capital_gain = st.number_input("Ganancias de Capital ($)", 0, 100000, 0)
    # Aquí podrías añadir selectbox para ocupación o estado civil 

# 3. Lógica de Predicción
if st.button("Verificar mi perfil"):
    # Crear un DataFrame vacío con las columnas que el modelo espera
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # Llenar los campos numéricos
    input_df['age'] = age
    input_df['education.num'] = edu_num
    input_df['hours.per.week'] = hours
    input_df['capital.gain'] = capital_gain
    
    # Escalado (Usamos el scaler guardado)
    # Nota: El scaler espera todas las numéricas; si no las pedimos, ponemos 0
    num_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    # Predicción
    probabilidad = rf_model.predict_proba(input_df)[0][1]
    
    # 4. Resultados Visuales
    st.markdown("### Resultado del Análisis")
    if probabilidad > 0.5:
        st.success(f"Tu probabilidad de ganar >50K es del **{probabilidad:.2%}**")
        st.balloons()
    else:
        st.warning(f"Tu probabilidad actual es del **{probabilidad:.2%}**")
        
    # Recomendación Dinámica
    st.info("💡 **Hoja de ruta recomendada:**")
    if edu_num < 13:
        st.write("- Considera completar un grado universitario (Bachelors). Históricamente, es el factor que más impulsa el ingreso en este modelo.")
    if hours < 40:
        st.write("- Incrementar tus horas semanales podría acercarte al umbral de éxito financiero.")
