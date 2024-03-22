import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
import h5py

# Carregar o modelo treinado

# Abra o arquivo diretamente com uma codificação específica
with h5py.File('model.h5', 'r', driver='core') as f:
    # Carregue o modelo
    model = load_model(f, compile=False)

label_to_text =  {'bus': 0 , 'bank': 1 , 'car' : 2 , 'formation': 3 , 'hospital' : 4 ,'I' : 5 , 'man' : 6 , 'motorcycle' : 7 , 'my' : 8 , 'supermarket' : 9 , 'we' : 10 ,
             'woman'  : 11 , 'you' : 12 , 'you (plural)' : 13  , 'your' : 14 }

# Função para processar o frame de vídeo e fazer a previsão do objeto
def predict_object(frame):
    # Redimensionar o frame para o tamanho esperado pelo modelo
    resized_frame = cv2.resize(frame, (48, 48))
    
    # Converter o frame para escala de cinza
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    
    # Normalizar os pixels da imagem
    normalized_frame = gray_frame.astype('float32') / 255.0
    
    # Pré-processar o frame para o modelo
    preprocessed_frame = normalized_frame.reshape(1, 48, 48, 1)
    
    # Fazer a previsão
    predicted_class = final_model_mlp.predict(preprocessed_frame).argmax()
    predicted_object = label_to_text[predicted_class]
    
    return predicted_object

# Função para reconhecer o objeto em uma imagem
def recognize_object(image):
    # Fazer a previsão do objeto
    predicted_object = predict_object(image)
    return predicted_object

# Função para redimensionar a imagem
def resize_image(image, target_size):
    # Redimensionar a imagem
    resized_image = cv2.resize(image, target_size)
    
    return resized_image

# Definir a classe do transformador de vídeo
class VideoTransformer(VideoTransformerBase):
    def _init_(self):
        # Inicializar o modelo de detecção de mãos
        self.hand_detection = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
    def transform(self, frame):
        # Converter o frame para um array numpy
        frame_bgr = np.array(frame.to_image())

        # Converter de BGR para RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Detectar mãos no frame
        results = self.hand_detection.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extrair coordenadas da mão
                hand_points = np.array([[lmk.x, lmk.y] for lmk in hand_landmarks.landmark]).astype(np.float32)
                xmin, ymin = np.min(hand_points, axis=0)
                xmax, ymax = np.max(hand_points, axis=0)
                xmin, ymin, xmax, ymax = int(xmin * frame_rgb.shape[1]), int(ymin * frame_rgb.shape[0]), \
                                         int(xmax * frame_rgb.shape[1]), int(ymax * frame_rgb.shape[0])
                
                # Recortar a região da mão
                hand_roi = frame_rgb[ymin:ymax, xmin:xmax]
                
                # Reconhecer o objeto (mão)
                predicted_object = recognize_object(hand_roi)
                
                # Desenhar a mão detectada no frame
                frame_rgb = cv2.putText(frame_rgb, predicted_object, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                print(predicted_object)
        
        return frame_rgb

#exibir imagem
st.sidebar.image("https://www.mjvinnovation.com/wp-content/uploads/2021/07/mjv_blogpost_redes_neurais_ilustracao_cerebro-01-1024x1020.png")
# Definir o título da página
st.sidebar.title('Reconhecimento de :red[Sinais] :ok_hand:')

# Breve descrição do projeto
st.sidebar.info("""
## Reconhecimento de Mãos - Projeto

Este projeto visa desenvolver um programa capaz de utilizar uma rede neural treinada para detectar mãos em tempo real por meio da câmera do usuário. Usando um modelo de rede neural CNN. 
""")

# Iniciar o streamlit webrtc
webrtc_streamer(key="hand-recognition-1", video_processor_factory=VideoTransformer)

# Integrantes do projeto
st.sidebar.write("""
## Integrantes

- Lorrayne
- Libhinny
- Samira
- Ytalo
""")