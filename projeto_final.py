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
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def get_emotion(face_landmarks, w, h):
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
    cor_alerta = (150, 150, 150)
    
    if abertura_boca > 0.30 and altura_sobrancelha > 0.50:
        emocao = "Surpresa/Choque"
        cor_alerta = (0, 255, 255)
        
    elif largura_boca < 0.65 and abertura_boca < 0.05 and distancia_sobrancelhas < 0.30:
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
frames_alegria = 0
frames_tensao = 0
frames_surpresa = 0
frames_neutro = 0

print("A INICIAR SESSÃO DE MONITORIZAÇÃO DE UX...")

while True:
    ret, frame = cap.read()
    if not ret: 
        break
        
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    largura_dash = 400
    dashboard = np.zeros((h, largura_dash, 3), dtype=np.uint8)
    dashboard[:] = (25, 25, 25) 
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int((time.time() - start_time) * 1000)
    
    results = landmarker.detect_for_video(mp_image, timestamp_ms)
    
    emocao_atual = "A aguardar Rosto..."
    cor = (150, 150, 150)
    lag_boca, abert_boca, dist_sob = 0, 0, 0
    
    if results.face_landmarks:
        face_landmarks = results.face_landmarks[0]
        frames_processados += 1
        
        emocao_atual, cor, lag_boca, abert_boca, dist_sob, pts_desenho = get_emotion(face_landmarks, w, h)
        
        if emocao_atual == "Alegria/Diversao":
            frames_alegria += 1
        elif emocao_atual == "Tensao/Foco":
            frames_tensao += 1
        elif emocao_atual == "Surpresa/Choque":
            frames_surpresa += 1
        else:
            frames_neutro += 1

        for pt in pts_desenho:
            cv2.circle(frame, pt, 3, cor, -1)
            
    
        pct_alegria = (frames_alegria / frames_processados) * 100
        pct_tensao = (frames_tensao / frames_processados) * 100
        pct_surpresa = (frames_surpresa / frames_processados) * 100
        pct_neutro = (frames_neutro / frames_processados) * 100
            

        cv2.putText(dashboard, "DASHBOARD UX", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        cv2.line(dashboard, (20, 55), (largura_dash - 20, 55), (100, 100, 100), 1)
        
        cv2.putText(dashboard, "ESTADO ATUAL:", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(dashboard, emocao_atual, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.1, cor, 2)
        
        cv2.line(dashboard, (20, 160), (largura_dash - 20, 160), (50, 50, 50), 1)
        
        cv2.putText(dashboard, "PREVALENCIA NA SESSAO (%):", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        cv2.putText(dashboard, f"Alegria: {pct_alegria:.1f}%", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(dashboard, (20, 230), (20 + int((pct_alegria/100)*340), 245), (0, 255, 0), -1) 

        cv2.putText(dashboard, f"Tensao: {pct_tensao:.1f}%", (20, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(dashboard, (20, 285), (20 + int((pct_tensao/100)*340), 300), (0, 0, 255), -1) 

        cv2.putText(dashboard, f"Surpresa: {pct_surpresa:.1f}%", (20, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(dashboard, (20, 340), (20 + int((pct_surpresa/100)*340), 355), (0, 255, 255), -1) 
        
        cv2.putText(dashboard, f"Neutro: {pct_neutro:.1f}%", (20, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(dashboard, (20, 395), (20 + int((pct_neutro/100)*340), 410), (150, 150, 150), -1) 
        
        cv2.line(dashboard, (20, 430), (largura_dash - 20, 430), (50, 50, 50), 1)
        
        cv2.putText(dashboard, f"DADOS RAW -> Boca: {lag_boca:.2f} | Abert: {abert_boca:.2f} | Sob: {dist_sob:.2f}", (20, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
        cv2.putText(dashboard, f"Frames analisados: {frames_processados}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    tela_final = np.hstack((frame, dashboard))

    cv2.imshow("Monitor de Engajamento UX", tela_final)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


if frames_processados > 0:
    print("\n" + "="*55)
    print(" RELATÓRIO DA SESSÃO DE UTILIZADOR (MÉTRICAS DE UX)")
    print("="*55)
    print(f"Total de Frames Analisados: {frames_processados}")
    print(f"- Neutro:   {pct_neutro:.1f}%")
    print(f"- Alegria:  {pct_alegria:.1f}%")
    print(f"- Tensao:   {pct_tensao:.1f}%")
    print(f"- Surpresa: {pct_surpresa:.1f}%")
    print("-" * 55)
    
    maior_emocao = max([pct_alegria, pct_tensao, pct_surpresa])
    
    if maior_emocao == pct_alegria and pct_alegria > 15:
        print("Insight: O conteúdo testado gerou forte resposta positiva.")
    elif maior_emocao == pct_tensao and pct_tensao > 15:
        print("Insight: O utilizador demonstrou forte foco/tensão (típico de jogos desafiantes).")
    elif maior_emocao == pct_surpresa and pct_surpresa > 10:
        print("Insight: O conteúdo gerou picos significativos de surpresa/choque.")
    else:
        print("Insight: Resposta global neutra. Necessita de validação com questionários de UX.")
    print("="*55 + "\n")

cap.release()
cv2.destroyAllWindows()