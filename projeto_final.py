import cv2
import numpy as np
import math
import time
import os
import urllib.request
import mediapipe as mp


model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    print("A transferir o ficheiro de modelo face_landmarker.task...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Transferência concluída!")


def dist(p1, p2):
    """Calcula a distância Euclidiana entre dois pontos (x, y)"""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def get_emotion(face_landmarks, w, h):
    """
    Analisa a geometria do rosto para inferir a emoção predominante.
    """

    boca_esq = (int(face_landmarks[61].x * w), int(face_landmarks[61].y * h))
    boca_dir = (int(face_landmarks[291].x * w), int(face_landmarks[291].y * h))
    boca_topo = (int(face_landmarks[13].x * w), int(face_landmarks[13].y * h))
    boca_base = (int(face_landmarks[14].x * w), int(face_landmarks[14].y * h))
    
  
    sob_esq_int = (int(face_landmarks[55].x * w), int(face_landmarks[55].y * h))
    sob_dir_int = (int(face_landmarks[285].x * w), int(face_landmarks[285].y * h))
    sob_esq_topo = (int(face_landmarks[105].x * w), int(face_landmarks[105].y * h))
    
   
    nariz = (int(face_landmarks[1].x * w), int(face_landmarks[1].y * h))
    queixo = (int(face_landmarks[152].x * w), int(face_landmarks[152].y * h))
    tamanho_rosto = dist(nariz, queixo)
    
  
    if tamanho_rosto == 0: 
        return "Neutro", (200, 200, 200), 0, 0, 0, []

   
    largura_boca = dist(boca_esq, boca_dir) / tamanho_rosto
    abertura_boca = dist(boca_topo, boca_base) / tamanho_rosto
    distancia_sobrancelhas = dist(sob_esq_int, sob_dir_int) / tamanho_rosto
    altura_sobrancelha = dist(sob_esq_topo, nariz) / tamanho_rosto

  
    emocao = "Neutro"
    cor_alerta = (200, 200, 200)
    
    
    if abertura_boca > 0.30 and altura_sobrancelha > 0.50:
        emocao = "Surpresa/Choque"
        cor_alerta = (0, 255, 255) 
        
    
    elif abertura_boca < 0.30 and distancia_sobrancelhas < 0.30:
        emocao = "Tensao/Foco"
        cor_alerta = (0, 0, 255) 
        
    
    elif largura_boca > 0.65 and abertura_boca < 0.32:
        emocao = "Alegria/Diversao"
        cor_alerta = (0, 255, 0)
        
    pontos_desenho = [boca_esq, boca_dir, boca_topo, boca_base, sob_esq_int, sob_dir_int]
    return emocao, cor_alerta, largura_boca, abertura_boca, distancia_sobrancelhas, pontos_desenho


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
landmarker = FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: Não foi possível aceder à webcam.")
    exit()

start_time = time.time()


frames_processados = 0
tempo_sorrindo = 0

print("A INICIAR SESSÃO DE MONITORIZAÇÃO DE UX...")

while True:
    ret, frame = cap.read()
    if not ret: 
        break
        
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int((time.time() - start_time) * 1000)
    
 
    results = landmarker.detect_for_video(mp_image, timestamp_ms)
    
   
    emocao_atual = "A aguardar Rosto..."
    cor = (255, 255, 255)
    lag_boca, abert_boca, dist_sob = 0, 0, 0
    
    if results.face_landmarks:
        face_landmarks = results.face_landmarks[0]
        frames_processados += 1
        
       
        emocao_atual, cor, lag_boca, abert_boca, dist_sob, pts_desenho = get_emotion(face_landmarks, w, h)
        
      
        if emocao_atual == "Alegria/Diversao":
            tempo_sorrindo += 1

        
        for pt in pts_desenho:
            cv2.circle(frame, pt, 3, cor, -1)
            
        
        cv2.rectangle(frame, (0, 0), (w, 130), (30, 30, 30), -1) 
        cv2.putText(frame, f"Emocao: {emocao_atual}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)
        
        
        cv2.putText(frame, f"Largura Boca (Sorriso): {lag_boca:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Abertura Boca (Surpresa): {abert_boca:.2f}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Dist. Sobrancelhas (Tensao): {dist_sob:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    
    cv2.imshow("UX Research: Monitorizacao Emocional", frame)
    
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if frames_processados > 0:
    pct_alegria = (tempo_sorrindo / frames_processados) * 100
    print("\n" + "="*55)
    print(" RELATÓRIO DA SESSÃO DE UTILIZADOR (MÉTRICAS DE UX)")
    print("="*55)
    print(f"Total de Frames Analisados: {frames_processados}")
    print(f"Taxa de Engajamento Positivo (Alegria): {pct_alegria:.1f}%")
    print("-" * 55)
    if pct_alegria > 30:
        print("Insight: O conteúdo testado gerou forte resposta positiva.")
    elif pct_alegria > 10:
        print("Insight: Resposta neutra com picos de engajamento.")
    else:
        print("Insight: O conteúdo não estimulou reações positivas significativas. Sugere-se revisão do design.")
    print("="*55 + "\n")

cap.release()
cv2.destroyAllWindows()