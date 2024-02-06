import cv2
import streamlit as st
import numpy as np


cap = cv2.VideoCapture(0)  # 0 representa a câmera padrão (pode variar dependendo do sistema)
placeholder = st.empty()

while True:
    ret, frame = cap.read()

    # Exibir frame no Streamlit
    if not ret:
        st.warning("Erro ao capturar o frame. Verifique a câmera.")
        break

    # Converter o frame para o formato RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Atualizar o contêiner vazio com o novo frame
    placeholder.image(rgb_frame, channels="RGB")

    # Verificar se o botão "Parar" foi pressionado
    if st.button("Parar"):
        break

# Liberar a câmera após o término do loop
cap.release()
