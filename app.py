# import streamlit as st
# import tensorflow as tf
# import os
# import pandas as pd

# # from reconhecimento import Ligar_webcam
# # from dataset import extrair_e_exibir_imagens


# # from ReconhecimentoDeSinais.main import extrair_e_exibir_imagens
# ## from modelo_treinamento import

# # setando o layout da pagina
# st.set_page_config(layout="wide")

# # setup sidebar

# with st.sidebar:
#     st.image(
#         "https://cdn.sanity.io/images/tlr8oxjg/production/ada93729daf922ad0318c8c0295e5cb477921808-1456x816.png?w=3840&q=100&fit=clip&auto=format"
#     )
#     st.header("Redes Neurais")
#     st.title("Reconhecimento da Linguagem de Sinais Brasileira")
#     info = """
#     Este trabalho utiliza técnicas avançadas de Machine Learning para criar um sistema capaz de
#     reconhecer e interpretar a Linguagem Brasileira de Sinais (Libras). A abordagem envolve a
#     coleta de dados multimodais, incluindo vídeos de sinais realizados por usuários fluentes em
#     Libras. Utilizando algoritmos de visão computacional e redes neurais, o sistema aprende a
#     mapear gestos específicos para suas correspondentes expressões na linguagem de sinais. Os
#     resultados promissores indicam um avanço significativo na comunicação inclusiva para surdos
#     e na aplicação de tecnologias de Machine Learning para problemas sociais relevantes.
#     """

#     st.info(info)

#     # função de capturar o video

#     # st.write("Clique no botão para exibir a webcam")
#     # if st.button("Inicie a webcam"):
#     #     Ligar_webcam()

#     # exibir_imagem = extrair_e_exibir_imagens()
#     # st.write("Imagens do dataset")
#     # st.image(exibir_imagem)

#     # @st.cache_resource
#     # def load_model():
#     #     #logica do modelo quando a mesma estiver pronta
#     #     model.eval()

#     #     return Model

#     # model = load_model()

#     import cv2
# import mediapipe as mp

# webcam = cv2.VideoCapture()
# ip = "https://192.168.1.5:8080/video"  # aqui é o ip do celular
# webcam.open(ip)  # para detectar a webcam

# hand = mp.solutions.hands.Hands(max_num_hands=2)  # mapear as mãos
# mpDraw = mp.solutions.drawing_utils  # desenhar as ligações entre os pontos das mãos

# try:
#     while True:
#         check, image = webcam.read()
#         if not check:
#             print("Erro ao capturar a imagem da webcam.")
#             break
#             # passar o parametro de imagem
#         imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = hand.process(imgRGB)

#         altura, largura, _ = image.shape

#         handsPoint = results.multi_hand_landmarks
#         if handsPoint:
#             for points in handsPoint:
#                 mpDraw.draw_landmarks(
#                     image, points, mp.solutions.hands.HAND_CONNECTIONS
#                 )

#                 for id, cord in enumerate(points.landmark):
#                     cx, cy = int(cord.x * largura), int(cord.y * altura)
#                     cv2.putText(
#                         image,
#                         str(id),
#                         (cx, cy + 10),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.5,
#                         (255, 0, 0),
#                         2,
#                     )

#         cv2.imshow("Imagem webcam", image)
#         key = cv2.waitKey(1)
#         if key == 27:  # ESC
#             break
# finally:
#     webcam.release()
#     cv2.destroyAllWindows()


import streamlit as st
import cv2
import mediapipe as mp

# Configuração da página
st.set_page_config(layout="wide")
st.sidebar.image(
    "https://cdn.sanity.io/images/tlr8oxjg/production/ada93729daf922ad0318c8c0295e5cb477921808-1456x816.png?w=3840&q=100&fit=clip&auto=format"
)
st.sidebar.header("Redes Neurais")
st.sidebar.title("Reconhecimento da Linguagem de Sinais Brasileira")
info = """
#     Este trabalho utiliza técnicas avançadas de Machine Learning para criar um sistema capaz de
#     reconhecer e interpretar a Linguagem Brasileira de Sinais (Libras). A abordagem envolve a
#     coleta de dados multimodais, incluindo vídeos de sinais realizados por usuários fluentes em
#     Libras. Utilizando algoritmos de visão computacional e redes neurais, o sistema aprende a
#     mapear gestos específicos para suas correspondentes expressões na linguagem de sinais. Os
#     resultados promissores indicam um avanço significativo na comunicação inclusiva para surdos
#     e na aplicação de tecnologias de Machine Learning para problemas sociais relevantes.
#     """

#     st.info(info)


# função da webcam
def ligar_webcam(ip):
    webcam = cv2.VideoCapture(ip)
    hand = mp.solutions.hands.Hands(max_num_hands=2)
    mpDraw = mp.solutions.drawing_utils
    # laço que verifica a webcam
    while True:
        check, image = webcam.read()
        if not check:
            st.error("Erro ao capturar a imagem da webcam.")
            break

        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hand.process(imgRGB)

        altura, largura, _ = image.shape

        handsPoint = results.multi_hand_landmarks
        if handsPoint:
            for points in handsPoint:
                mpDraw.draw_landmarks(
                    image, points, mp.solutions.hands.HAND_CONNECTIONS
                )

                for id, cord in enumerate(points.landmark):
                    cx, cy = int(cord.x * largura), int(cord.y * altura)
                    cv2.putText(
                        image,
                        str(id),
                        (cx, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2,
                    )

        st.image(image, channels="BGR")

    webcam.release()
    cv2.destroyAllWindows()


# Sidebar do ip (LORRAYNE SEM WEBCAM POR ISSO USANDO O IP)
ip = st.sidebar.number_input("Insira o IP da webcam 0 para frontal e 1 para traseira:",step=1)

# iniciar a Webcam
if st.sidebar.button("Iniciar Webcam"):
    ligar_webcam(ip)
