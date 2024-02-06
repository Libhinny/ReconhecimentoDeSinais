import cv2
import mediapipe as mp


webcam = cv2.VideoCapture(0)
hand = mp.solutions.hands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

def Ligar_webcam():
    
    try:
        while True:
            check, image = webcam.read()
            if not check:
                print("Erro ao capturar a imagem da webcam.")
                break

            imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hand.process(imgRGB)

            altura, largura, _ = image.shape

            handsPoint = results.multi_hand_landmarks
            if handsPoint:
                for points in handsPoint:
                    mpDraw.draw_landmarks(image, points, mp.solutions.hands.HAND_CONNECTIONS)

                    for id, cord in enumerate(points.landmark):
                        cx, cy = int(cord.x * largura), int(cord.y * altura)
                        cv2.putText(image, str(id), (cx, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.imshow("Imagem da sua webcam", image)
            key = cv2.waitKey(5)
            if key == 27:  # ESC
                break
    finally:
        webcam.release()
        cv2.destroyAllWindows()