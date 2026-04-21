import cv2
import numpy as np


def main():
    cap = cv2.VideoCapture(0) ## fonte de vídeo (0 para webcam padrão) ou pode ser um arquivo de vídeo, por exemplo: 'video.mp4', lembra de ajustar o path
    if not cap.isOpened():
        print("Erro: Não foi possível acessar a webcam")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível capturar o frame")
            break
        
        # Aqui você pode adicionar processamento de imagem
        frame = cv2.flip(frame, 1)  # espelha o frame horizontalmente
        
        
        
        # aqui vc exibe o frame processado
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): # Pressione 'q' para sair
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
