import cv2
import sys
import time
import os # Necesario para verificar la existencia del archivo de video

# --- Configuración ---
#C:/Users/funky/Documents/vision por computador/Matching/Mosaico1/Fotos/foto1 
# Cambia esto a la ruta de tu archivo de video "C:\Users\funky\Documents\vision por computador\videossss\videoprueba.mp4"
#"            C:\Users\funky\OneDrive\Documentos\vision por computador\videossss\video.mp4"
#"C:\Users\funky\OneDrive\Documentos\vision por computador\videossss\video.mp4"
video_path = 'C:/Users/funky/OneDrive/Documentos/vision por computador/videossss/video.mp4'
# -------------------

# Verifica si el archivo de video existe
if not os.path.exists(video_path):
    print(f"Error: El archivo de video '{video_path}' no fue encontrado.")
    sys.exit()

# ¡SOLO COMO ÚLTIMO RECURSO SI LA INSTALACIÓN DE CONTRIB NO FUNCIONA!
tracker_types = {
    'BOOSTING': cv2.legacy.TrackerBoosting_create,
    'MIL': cv2.TrackerMIL_create,
    'KCF': cv2.TrackerKCF_create,
    'TLD': cv2.legacy.TrackerTLD_create,
    'MOSSE': cv2.legacy.TrackerMOSSE_create,
    'CSRT': cv2.TrackerCSRT_create
}
# Crear instancias de todos los trackers
trackers = {}
for tracker_name, tracker_create_func in tracker_types.items():
    trackers[tracker_name] = tracker_create_func()

# Leer video
video = cv2.VideoCapture(video_path)

# Verificar si el video se abrió correctamente
if not video.isOpened():
    print("Error al abrir el archivo de video.")
    sys.exit()

# Leer el primer frame
ok, frame = video.read()
if not ok:
    print("No se pudo leer el video.")
    sys.exit()

# Seleccionar el Bounding Box inicial (ROI - Region of Interest)
# Se usará el mismo ROI para inicializar todos los trackers
print("Selecciona el objeto a seguir y presiona ESPACIO o ENTER.")
print("Presiona 'c' para cancelar la selección.")
bbox = cv2.selectROI("Selecciona ROI", frame, False)
cv2.destroyWindow("Selecciona ROI") # Cerrar la ventana de selección

if not bbox or bbox == (0, 0, 0, 0):
    print("No se seleccionó ningún objeto. Saliendo.")
    sys.exit()

print(f"ROI inicial seleccionado: {bbox}")

# Inicializar cada tracker con el primer frame y el ROI
initialization_success = {}
for name, tracker in trackers.items():
    try:
        # Algunos trackers pueden fallar en la inicialización si el ROI es inválido
        # o si hay problemas con la versión de OpenCV o sus dependencias.
        ok_init = tracker.init(frame, bbox)
        initialization_success[name] = ok_init
        if not ok_init:
             print(f"Fallo al inicializar el tracker: {name}")
        else:
             print(f"Tracker {name} inicializado correctamente.")
    except Exception as e:
        print(f"Excepción al inicializar {name}: {e}")
        initialization_success[name] = False
        # Remover el tracker que falló para no intentar actualizarlo
        # Es importante manejar esto para que el bucle principal no falle
        # trackers.pop(name) # Podríamos removerlo, pero es mejor marcarlo como fallido



# Diccionario para almacenar resultados (FPS, éxito por frame)
results = {name: {'fps': [], 'success': []} for name in trackers if initialization_success.get(name, False)}

# Colores para dibujar los bounding boxes de cada tracker
# Genera colores distintos (BGR)
colors = {
    'BOOSTING': (255, 0, 0),   # Azul
    'MIL': (0, 255, 0),     # Verde
    'KCF': (0, 0, 255),     # Rojo
    'TLD': (255, 255, 0),   # Cyan
    'MOSSE': (255, 0, 255),   # Magenta
    'CSRT': (0, 255, 255)    # Amarillo
}

frame_count = 0

while True:
    # Leer un nuevo frame
    ok_frame, frame = video.read()
    if not ok_frame:
        break # Fin del video

    frame_count += 1
    display_frame = frame.copy() # Crear una copia para dibujar sobre ella

    # Actualizar cada tracker y medir el tiempo
    for name, tracker in trackers.items():
        # Solo actualizar si la inicialización fue exitosa
        if not initialization_success.get(name, False):
            continue

        start_time = time.time()
        ok_update, new_bbox = tracker.update(frame)
        end_time = time.time()

        # Calcular FPS para este frame y este tracker
        fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0

        # Almacenar resultados
        results[name]['fps'].append(fps)
        results[name]['success'].append(ok_update)

        # Dibujar bounding box y etiqueta en el frame de visualización
        if ok_update:
            p1 = (int(new_bbox[0]), int(new_bbox[1]))
            p2 = (int(new_bbox[0] + new_bbox[2]), int(new_bbox[1] + new_bbox[3]))
            cv2.rectangle(display_frame, p1, p2, colors[name], 2, 1)
            label = f"{name}: OK ({fps:.1f} FPS)"
            cv2.putText(display_frame, label, (p1[0], p1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[name], 1)
        else:
            # Indicar fallo del track111er
            label = f"{name}: FAILED"
            # Poner el texto en una esquina o posición fija si falla
            # (Aquí lo ponemos cerca de la última posición conocida si existiera,
            # o en una esquina si falla desde el principio)
            # Para simplificar, lo ponemos en una posición fija superior.
            text_y_pos = list(trackers.keys()).index(name) * 20 + 20
            cv2.putText(display_frame, label, (10, text_y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[name], 1)

    # Mostrar el frame con todos los resultados
    cv2.imshow("Tracking Comparison", display_frame)

    # Salir si se presiona la tecla ESC
    k = cv2.waitKey(1) & 0xff
    if k == 27: # ESC
        break

# Liberar recursos
video.release()
cv2.destroyAllWindows()

# --- Procesamiento Básico de Resultados ---
print("\n--- Resumen de Resultados ---")
if frame_count > 0:
    for name in results:
        if not results[name]['fps']: # Si no hubo frames procesados para este tracker
            print(f"\nTracker: {name}")
            print("  No se procesaron frames (posible fallo de inicialización).")
            continue

        avg_fps = sum(results[name]['fps']) / len(results[name]['fps'])
        success_rate = (sum(results[name]['success']) / frame_count) * 100 # Éxito sobre frames procesados
        print(f"\nTracker: {name}")
        print(f"  FPS Promedio: {avg_fps:.2f}")
        print(f"  Tasa de Éxito (frames donde se encontró el objeto): {success_rate:.2f}%")
else:
    print("No se procesaron frames del video.")

print("\nNota: La 'Tasa de Éxito' indica si el tracker reportó haber encontrado el objeto,")
print("no necesariamente si la posición es correcta (eso requeriría ground truth).")