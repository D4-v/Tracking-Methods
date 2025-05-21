import cv2

# Diccionario con los trackers disponibles en OpenCV
TRACKERS = {
    'BOOSTING': cv2.legacy.TrackerBoosting_create,
    'MIL': cv2.TrackerMIL_create,
    'KCF': cv2.TrackerKCF_create,
    'TLD': cv2.legacy.TrackerTLD_create,
    'MOSSE': cv2.legacy.TrackerMOSSE_create,
    'CSRT': cv2.TrackerCSRT_create
}

def create_tracker(tracker_type):
    if tracker_type in TRACKERS:
        return TRACKERS[tracker_type]()
    else:
        raise ValueError(f"Tracker '{tracker_type}' no es válido.")



def main():
    cap = cv2.VideoCapture("C:/Users/funky/Documents/vision por computador/videossss/video.mp4")  # Captura desde la cámara (puedes cambiarlo a un video)
    if not cap.isOpened():
        print("Error el video")
        return
    
    tracker_type = 'CSRT'  # Cambia esto para probar otros métodos
    tracker = create_tracker(tracker_type)
    
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el frame inicial")
        cap.release()
        return
    
    bbox = cv2.selectROI("Selecciona el objeto", frame, False)
    tracker.init(frame, bbox)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        success, bbox = tracker.update(frame)
        
        if success:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
        else:
            cv2.putText(frame, "Tracking fallido", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        
        cv2.putText(frame, f"Tracker: {tracker_type}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        cv2.imshow("Tracking", frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # Tecla 'Esc' para salir
            break
    
    cap.release()
    
    cv2.destroyAllWindows()
    




if __name__ == "__main__":
    main()
    
    input_path = "C:/Users/funky/Documents/vision por computador/videossss/video.mp4"
    #output_path = "C:/Users/funky/Documents/vision por computador/videossss/Videosf/KCF.mp4"
    #save_tracking_video(input_path, output_path, tracker_type='KCF')