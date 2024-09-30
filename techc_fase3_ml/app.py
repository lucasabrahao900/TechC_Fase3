import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import random

import warnings
warnings.filterwarnings("ignore")

def cria_dummy(var, cat_ref):
    if var == cat_ref:
        return 0
    else:
        return 1

# Load the trained Random Forest model
with open(r'C:\Users\lucaa\Desktop\Pastas Gerais\Pós Tech Data Analytics\Tech Challenge\TechChallenge - Fase 3\techc_fase3_ml\models\random_forest_model.pkl', 'rb') as file:
    loaded_rf_model = pickle.load(file)

# Streamlit app title
st.title("Acompanhamento Auxiliar Cardiológico - Analytics App")

# Descrição do App:
st.write("Este APP tem o objetivo de auxiliar na prevenção de potenciais insuficiências cardíacas, visando tanto um acompanhamento contínuo de um paciente internado quanto em um diagnóstico único.")
st.write('''
Abaixo algumas características devem ser preenchidas que serão utilizadas tanto para um acompamnhento contínuo quanto para umda predição única.         
''')

# Input fields for a single prediction
idade = st.number_input("Idade", min_value=0, max_value=120, value=30, step=1)
sexo = st.selectbox("Sexo", ("Feminino", "Masculino"))
anemia = st.selectbox("Anemia", ("Sim", "Não"))
diabetes = st.selectbox("Diabetes", ("Sim", "Não"))
hg_bp = st.selectbox("Pressão Alta", ("Sim", "Não"))
fumante = st.selectbox("Fumante", ("Sim", "Não"))


# Define tabs (frames) for singular and batch (streaming) prediction
tab1, tab2 = st.tabs(["Predição Única", "Múltiplas Predições"])


# 1. Unica
with tab1:
    st.header("Predição Única")
    st.write('''
    Nesse formato, é necessário que os dados abaixo sejam preenchidos manualmente para realização de uma predição.
    ''')

    # Valores Numericos para Medições:
    creatina_fosfoquinase = st.number_input("Creatina Fosfoquinase", value=0, step=1)
    fracao_de_ejecao = st.number_input("Fração de Ejeção", value=0, step=1)
    plaquetas = st.number_input("Plaquetas", value=0, step=1)
    creatinina_serica = st.number_input("Creatinina Sérica", value=0, step=1)
    sodio_serico = st.number_input("Sódio Sérico", value=0, step=1)

    start_prediction_1 = st.button("Realizar Predição")

    if start_prediction_1:
        input_data = [
            float(idade),
            cria_dummy(var = anemia, cat_ref = "Não"),
            int(creatina_fosfoquinase),
            cria_dummy(var = diabetes, cat_ref = "Não"),
            int(fracao_de_ejecao),
            cria_dummy(var = hg_bp, cat_ref = "Não"),
            float(plaquetas),
            float(creatinina_serica),
            int(sodio_serico),
            cria_dummy(var = sexo, cat_ref = "Feminino"),
            cria_dummy(var = fumante, cat_ref = "Não")
        ]

        # Realizando predição unica:
        prob_insuf = loaded_rf_model.predict_proba([input_data])[0][1]

        st.write(f"A probabilidade de Insuficiência Cardíaca do paciente é de {prob_insuf:.2%}.")


    

# 1. Streaming
with tab2:
    st.header("Multíplas Predições")
    st.write('''
    Nesse formato, é necessário que apenas um intervalo de tempo seja definido e os dados serão coletados a cada X tempo para uma predição, plotando todas as informações em tela.
    ''')
    tempo_minutos = st.number_input("Intervalo (em minutos)", value=0, step=1)

    start_prediction_2 = st.button("Iniciar Acompanhamento")
    plot_container = st.empty()
    df_prob = pd.DataFrame(columns=['time', 'prob_insuf'])

    if start_prediction_2:
        iteration = 0  # Track the iteration number
        start_time = time.time()
        flag = True
        while flag:

            # Valores Numericos para Medições:
            creatina_fosfoquinase = random.randint(23, 8000)
            fracao_de_ejecao = random.randint(14, 80)
            plaquetas = random.uniform(25100, 850000)
            creatinina_serica = random.uniform(0.5, 9.4)
            sodio_serico = random.randint(113, 148)
            
            
            input_data = [
                float(idade),
                cria_dummy(var = anemia, cat_ref = "Não"),
                int(creatina_fosfoquinase),
                cria_dummy(var = diabetes, cat_ref = "Não"),
                int(fracao_de_ejecao),
                cria_dummy(var = hg_bp, cat_ref = "Não"),
                float(plaquetas),
                float(creatinina_serica),
                int(sodio_serico),
                cria_dummy(var = sexo, cat_ref = "Feminino"),
                cria_dummy(var = fumante, cat_ref = "Não")
            ]

            # Record the current time (elapsed time in minutes)
            current_time = (time.time() - start_time) / 60  # Time in minutes
            prob_insuf = loaded_rf_model.predict_proba([input_data])[0][1]
            df_prob = pd.concat([df_prob, pd.DataFrame({'time': [current_time], 'prob_insuf': [prob_insuf]})]).reset_index(drop = True)

            # Plot the time series
            plt.figure(figsize=(10, 5))
            plt.plot(df_prob['time'], df_prob['prob_insuf'], marker='o', linestyle='-', color='orange')
            plt.title("Probabilidade de Insuficiência Cardíaca do Paciente")
            plt.xlabel("Tempo (em minutos)")
            plt.ylabel("Probabilidade de Insuficiência")
            plt.axhline(y = 0.5, color='r', linestyle='--', label='Prob. 50%')
            plt.ylim(0, 1)
            plt.grid(True)

            # Display the plot in the container
            plot_container.pyplot(plt)

            # Sleep for the given time interval (converted to seconds)
            iteration += 1

            if iteration > tempo_minutos:
                flag = False