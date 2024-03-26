import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import h5py
from mediapipe.python.solutions import hands

# Carregar o modelo treinado
with h5py.File(r'C:\Users\ytalo\OneDrive\Área de Trabalho\projeto redes neurais\ReconhecimentoDeSinais\checkpointCNN\model.h5', 'r', driver='core') as f:
    final_model_mlp = load_model(f, compile=False)


label_to_text = {
    0: 'bus',
    1: 'bank',
    2: 'car',
    3: 'formation',
    4: 'hospital',
    5: 'I',
    6: 'man',
    7: 'motorcycle',
    8: 'my',
    9: 'supermarket',
    10: 'we',
    11: 'woman',
    12: 'you',
    13: 'your',
    14: 'bus',
    15: 'you (plural)'
}

def predict_object(frame):
    # Converter o frame para uma imagem colorida (3 canais de cor)
    frame_color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    resized_frame = cv2.resize(frame_color, (120, 213))
    
    # Normalizar os pixels
    normalized_frame = resized_frame.astype('float32') / 255.0
    

    preprocessed_frame = normalized_frame.transpose((1, 0, 2))  # Inverter as dimensões (120, 213) para (213, 120)
    preprocessed_frame = preprocessed_frame.reshape(1, 213, 120, 3)
    
    # Fazer a previsão
    predicted_class = final_model_mlp.predict(preprocessed_frame).argmax()
    predicted_object = label_to_text[predicted_class]
    
    return predicted_object

# Função para reconhecer o objeto em uma imagem
def recognize_object(image):
    # Fazer a previsão do objeto
    predicted_object = predict_object(image)
    return predicted_object


hand_detection = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Capturar vídeo da câmera
cap = cv2.VideoCapture(0)

while True:
    # Ler o próximo frame
    ret, frame = cap.read()
    
    # Converter o frame para RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detectar mãos no frame
    results = hand_detection.process(frame_rgb)
    
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

            # Reconhecer a mao
            predicted_object = recognize_object(hand_roi)

            # Desenhar a mão detectada no frame
            frame = cv2.putText(frame, predicted_object, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    # Exibir o frame
    cv2.imshow('Hand Detection', frame)
    
    # q sai do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()