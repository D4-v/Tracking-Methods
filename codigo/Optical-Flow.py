import cv2
import numpy as np

cap = cv2.VideoCapture("C:/Users/funky/Documents/vision por computador/videossss/slow_traffic_small.mp4")

# Parámetros de detección de esquinas (Shi-Tomasi)
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parámetros de Optical Flow (Lucas-Kanade)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Colores aleatorios para cada punto de seguimiento
color = np.random.randint(0, 255, (100, 3))

# Capturar el primer frame
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Crear una máscara para dibujar las trayectorias
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcular Optical Flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Seleccionar los puntos válidos
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Dibujar las trayectorias
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

        img = cv2.add(frame, mask)
        cv2.imshow('Optical Flow (Lucas-Kanade)', img)

        # Actualizar el frame anterior y los puntos anteriores
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    # Salir con la tecla ESC
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()