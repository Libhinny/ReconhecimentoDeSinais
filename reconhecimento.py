import cv2
import mediapipe as mp

webcam = cv2.VideoCapture()
ip = "https://192.168.1.10:8080/video"  # aqui é o ip do celular
webcam.open(ip)  # para detectar a webcam

hand = mp.solutions.hands.Hands(max_num_hands=2)  # mapear as mãos
mpDraw = mp.solutions.drawing_utils  # desenhar as ligações entre os pontos das mãos

try:
    while True:
        check, image = webcam.read()
        if not check:
            print("Erro ao capturar a imagem da webcam.")
            break
            # passar o parametro de imagem
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

        cv2.imshow("Imagem webcam", image)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
finally:
    webcam.release()
    cv2.destroyAllWindows()


# import cv2


# if webcam.isOpened():
#     validacao, frame = webcam.read()
#     while validacao:
#         validacao, frame = webcam.read()
#         cv2.imshow("Video da Webcam", frame)
#         key = cv2.waitKey(1)
#         if key == 27:  # ESC
#             break
#     cv2.imwrite("FotoLira.png", frame)

# webcam.release()
# cv2.destroyAllWindows()
