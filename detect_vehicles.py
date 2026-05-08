"""
LPR - Etapa 1: Detección y Tracking de Vehículos
==================================================
Detecta vehículos en el stream RTSP, les asigna un ID
persistente (ByteTrack) y agrupa todos sus frames en una
carpeta por vehículo, listos para la Etapa 2.

Estructura de salida:
  output_stage1/
  ├── vehicles/
  │   ├── track_0001/
  │   │   ├── frame_0042.jpg
  │   │   ├── frame_0045.jpg
  │   │   └── metadata.json
  │   └── track_0002/
  │       └── ...
  └── index.json

ROI: Si existe config/roi.json, solo se procesan vehículos
     dentro de esa zona. Usar tools/define_roi.py para editar.
"""

import cv2
import json
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ===============================
# CONFIGURACION Y AJUSTES
# ===============================

# --- HISTORIAL DE CAMBIOS (Para referencia y reversión) ---
# 1. ORIG: Tracker=ByteTrack, Process=2, ConfTracker=0.25, ConfSave=0.80, Sharp=100
# 2. TEST: Tracker=ByteTrack, Process=1, ConfTracker=0.20, ConfSave=0.35, Sharp=40 (Duplicaba IDs)
# 3. NOW : Tracker=BoTSORT,   Process=1, ConfTracker=0.45, ConfSave=0.40, Sharp=40
# ----------------------------------------------------------

RTSP_URL   = "rtsp://admin:Muni2026@192.168.115.221:554/cam/realmonitor?channel=1&subtype=0"
MODEL_PATH = "yolov8n.pt"

PROCESS_EVERY = 1              # Salto de frames (1=todos, 2=uno por medio)
TRACKER_CONFIDENCE_THRESHOLD = 0.45  # Confianza para que el Tracker asigne ID
SAVE_CONFIDENCE_THRESHOLD    = 0.40  # Confianza para guardar el recorte

# Filtros de Calidad
MIN_BBOX_WIDTH = 50      
MIN_BBOX_HEIGHT = 50     
MIN_SHARPNESS = 40.0           # Nitidez (Laplaciano). Bajar si no guarda nada.
MIN_MOVEMENT_PIXELS = 5  
MAX_STATIONARY_FRAMES = 10 

# Expandir bbox SOLO para el recorte guardado (para incluir mejor el vehículo/patente)
# Recomendación: más padding hacia abajo (donde suele estar la patente).
# El cálculo de nitidez y filtros se hace con el bbox original.
BBOX_PADDING_RATIO_X = 0.15
BBOX_PADDING_RATIO_Y_TOP = 0.10
BBOX_PADDING_RATIO_Y_BOTTOM = 0.25
BBOX_PADDING_MIN_PIXELS = 8

# Clases de vehículos en COCO
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Directorios de salida
OUTPUT_DIR = Path("output_stage1")
ROI_CONFIG = Path("config/roi.json")

# ===============================
# PREPARAR DIRECTORIOS
# ===============================

vehicles_dir = OUTPUT_DIR / "vehicles"
vehicles_dir.mkdir(parents=True, exist_ok=True)

index_file = OUTPUT_DIR / "index.json"

# Cargar index existente
track_id_offset = 0
if index_file.exists():
    with open(index_file) as f:
        index = json.load(f)
    print(f"[INFO] Index existente: {len(index)} vehículos previos")
    for k in index.keys():
        if k.startswith("track_"):
            try:
                val = int(k.split("_")[1])
                if val > track_id_offset:
                    track_id_offset = val
            except:
                pass
else:
    index = {}

# ===============================
# CARGAR ROI (si existe)
# ===============================

roi_polygon = None

if ROI_CONFIG.exists():
    with open(ROI_CONFIG) as f:
        roi_data = json.load(f)
    roi_polygon = np.array(roi_data["roi"], dtype=np.int32)
    print(f"[INFO] ROI cargado: {len(roi_data['roi'])} puntos")
else:
    print("[WARN] Sin ROI definido — se detecta en todo el frame")
    print("       Ejecutar: python tools/define_roi.py")


def calculate_sharpness(image):
    """Calcula la varianza del Laplaciano para determinar la nitidez de la imagen."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def is_partially_in_roi(x1, y1, x2, y2):
    """Retorna True si cualquier esquina o el centro del bbox está dentro del ROI."""
    if roi_polygon is None:
        return True
    
    # Puntos a probar: 4 esquinas + centro
    points = [
        (x1, y1), (x2, y1), (x1, y2), (x2, y2),
        (int((x1+x2)/2), int((y1+y2)/2))
    ]
    
    for pt in points:
        if cv2.pointPolygonTest(roi_polygon, (float(pt[0]), float(pt[1])), False) >= 0:
            return True
    return False


def track_dir(track_id: int) -> Path:
    """Retorna la carpeta del vehículo, creándola si no existe."""
    d = vehicles_dir / f"track_{track_id:04d}"
    d.mkdir(exist_ok=True)
    return d


def expand_bbox_for_crop(x1: int, y1: int, x2: int, y2: int, frame_shape):
    """Expande bbox con padding (asimétrico en Y) y lo limita a los bordes del frame.

    Nota: pensado para el recorte guardado (mejor chance de incluir patente).
    """
    h_img, w_img = frame_shape[:2]
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    pad_x = max(int(round(w * BBOX_PADDING_RATIO_X)), BBOX_PADDING_MIN_PIXELS)
    pad_y_top = max(int(round(h * BBOX_PADDING_RATIO_Y_TOP)), BBOX_PADDING_MIN_PIXELS)
    pad_y_bottom = max(int(round(h * BBOX_PADDING_RATIO_Y_BOTTOM)), BBOX_PADDING_MIN_PIXELS)

    ex1 = max(0, x1 - pad_x)
    ey1 = max(0, y1 - pad_y_top)
    ex2 = min(w_img, x2 + pad_x)
    ey2 = min(h_img, y2 + pad_y_bottom)
    return ex1, ey1, ex2, ey2


def save_frame(track_id: int, frame_number: int, crop, bbox, conf, class_name, timestamp, sharpness):
    """Guarda el crop y actualiza metadata.json del vehículo."""
    folder = track_dir(track_id)
    track_key = f"track_{track_id:04d}"

    # Nombre del archivo del crop
    crop_filename = f"frame_{frame_number:06d}.jpg"
    cv2.imwrite(str(folder / crop_filename), crop)

    # Registro del frame
    frame_record = {
        "file":         crop_filename,
        "frame_number": frame_number,
        "timestamp":    timestamp,
        "confidence":   round(conf, 4),
        "sharpness":    round(sharpness, 2),
        "bbox":         {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]},
    }

    # Actualizar o crear metadata del vehículo
    meta_file = folder / "metadata.json"
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        meta["frames"].append(frame_record)
        meta["last_seen"]    = timestamp
        meta["total_frames"] = len(meta["frames"])
    else:
        meta = {
            "track_id":    track_id,
            "class_name":  class_name,
            "first_seen":  timestamp,
            "last_seen":   timestamp,
            "total_frames": 1,
            "frames":      [frame_record],
        }

    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    # Actualizar index global
    index[track_key] = {
        "track_id":    track_id,
        "class_name":  class_name,
        "first_seen":  meta["first_seen"],
        "last_seen":   timestamp,
        "total_frames": meta["total_frames"],
        "folder":      str(folder.relative_to(OUTPUT_DIR)),
    }
    with open(index_file, "w") as f:
        json.dump(index, f, indent=2)

    return crop_filename

# ===============================
# CARGAR MODELO
# ===============================

print("[INFO] Cargando modelo YOLO...")
model = YOLO(MODEL_PATH)
print(f"[INFO] Modelo '{MODEL_PATH}' listo")

# ===============================
# CONECTAR CAMARA
# ===============================

print(f"[INFO] Conectando a: {RTSP_URL.split('@')[1].split('/')[0]}")
cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("[ERROR] No se pudo conectar a la cámara RTSP")
    exit(1)

print("[INFO] Cámara conectada")

# ===============================
# VARIABLES DE ESTADO
# ===============================

frame_count   = 0
total_saves   = 0
prev_time     = time.time()
track_mapping = {}
next_track_id = track_id_offset + 1
vehicle_states = {}

# ===============================
# LOOP PRINCIPAL
# ===============================

print("\n[INFO] Iniciando detección con tracking...")
print("       Presionar 'Q' para salir\n")

while True:
    ret, frame = cap.read()

    if not ret:
        print("[WARN] Error leyendo frame — reconectando...")
        cap.release()
        time.sleep(2)
        cap = cv2.VideoCapture(RTSP_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        continue

    frame_count += 1

    if frame_count % PROCESS_EVERY != 0:
        continue

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    annotated = frame.copy()

    # --------------------------------------------------
    # INFERENCIA CON TRACKING (BoTSORT es más robusto contra duplicados)
    # --------------------------------------------------
    results = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml", # PREV: bytetrack.yaml
        verbose=False,
        conf=TRACKER_CONFIDENCE_THRESHOLD,
        classes=list(VEHICLE_CLASSES.keys()),
    )

    boxes    = results[0].boxes
    # Sin detecciones con ID asignado en este frame
    if boxes.id is None:
        # Mostrar frame sin anotaciones de tracking
        if roi_polygon is not None:
            cv2.polylines(annotated, [roi_polygon], True, (0, 255, 255), 2)
        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time
        cv2.putText(annotated, f"FPS: {fps:.1f}",        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(annotated, f"Frame: {frame_count}",   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(annotated, f"Vehiculos (totales): {len(index)}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(annotated, f"Guardados: {total_saves}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.imshow("LPR - Etapa 1: Tracking de Vehiculos", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    for box in boxes:
        if box.id is None:
            continue
            
        tid_raw = int(box.id[0])
        conf   = float(box.conf[0])
        cls_id = int(box.cls[0])

        x1, y1, x2, y2 = np.round(box.xyxy[0].cpu().numpy()).astype(int).tolist()
        w, h = x2 - x1, y2 - y1

        if tid_raw not in track_mapping:
            if cls_id in VEHICLE_CLASSES:
                track_mapping[tid_raw] = next_track_id
                next_track_id += 1
        
        tid = track_mapping.get(tid_raw, None)
        
        # --- VALIDACIÓN DE ROI ---
        is_inside_roi = is_partially_in_roi(x1, y1, x2, y2)
        
        status_msg = ""
        status_color = (0, 0, 255) # Rojo por defecto (ignorado)

        # --- FILTROS DE GUARDADO ---
        if tid is None: 
            status_msg = "BUSCANDO ID"
        elif not is_inside_roi: 
            status_msg = "FUERA ROI"
        elif conf < SAVE_CONFIDENCE_THRESHOLD: 
            status_msg = f"CONF BAJA ({conf:.2f})"
        elif w < MIN_BBOX_WIDTH or h < MIN_BBOX_HEIGHT: 
            status_msg = "MUY PEQUENO"
        else:
            # Crop "base" para estimar nitidez (bbox original)
            crop_for_sharpness = frame[y1:y2, x1:x2]
            if crop_for_sharpness.size == 0: 
                status_msg = "ERROR CROP"
            else:
                sharpness = calculate_sharpness(crop_for_sharpness)
                if sharpness < MIN_SHARPNESS:
                    status_msg = f"BORROSO ({sharpness:.0f})"
                else:
                    # Filtro de Movimiento
                    cx, cy = x1 + w // 2, y1 + h // 2
                    should_save = False
                    
                    if tid not in vehicle_states:
                        vehicle_states[tid] = {"last_center": (cx, cy), "stationary_count": 1}
                        should_save = True
                    else:
                        last_cx, last_cy = vehicle_states[tid]["last_center"]
                        dist = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
                        if dist < MIN_MOVEMENT_PIXELS:
                            vehicle_states[tid]["stationary_count"] += 1
                            if vehicle_states[tid]["stationary_count"] <= MAX_STATIONARY_FRAMES:
                                should_save = True
                            else:
                                status_msg = "ESTACIONARIO"
                        else:
                            vehicle_states[tid]["last_center"] = (cx, cy)
                            vehicle_states[tid]["stationary_count"] = 1
                            should_save = True

                    if should_save:
                        status_msg = "GUARDANDO..."
                        status_color = (0, 255, 0) # Verde
                        
                        crop_x1, crop_y1, crop_x2, crop_y2 = expand_bbox_for_crop(x1, y1, x2, y2, frame.shape)
                        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                        
                        if crop.size > 0:
                            fname = save_frame(
                                tid, frame_count, crop,
                                (crop_x1, crop_y1, crop_x2, crop_y2),
                                conf, VEHICLE_CLASSES[cls_id], timestamp, sharpness
                            )
                            total_saves += 1
                            # Log opcional en consola
                            # print(f"[SAVE] track={tid:04d} | {fname}")

        # --- DIBUJO EN PANTALLA ---
        if tid is not None and cls_id in VEHICLE_CLASSES:
            class_name = VEHICLE_CLASSES[cls_id]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), status_color, 2)
            label = f"ID:{tid} {class_name} {conf:.0%}"
            cv2.putText(annotated, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
            if status_msg:
                cv2.putText(annotated, status_msg, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)

    # --------------------------------------------------
    # DIBUJAR ROI Y UI
    # --------------------------------------------------
    if roi_polygon is not None:
        cv2.polylines(annotated, [roi_polygon], True, (0, 255, 255), 2)

    current_time = time.time()
    fps = 1.0 / max(current_time - prev_time, 1e-6)
    prev_time = current_time

    cv2.putText(annotated, f"FPS: {fps:.1f}",              (20, 35),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(annotated, f"Frame: {frame_count}",         (20, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(annotated, f"Vehiculos (totales): {len(index)}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(annotated, f"Frames guardados: {total_saves}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    roi_status = "ROI: ACTIVO" if roi_polygon is not None else "ROI: sin definir"
    roi_color  = (0, 255, 255) if roi_polygon is not None else (0, 100, 255)
    cv2.putText(annotated, roi_status, (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 1)

    cv2.imshow("LPR - Etapa 1: Tracking de Vehiculos", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\n[INFO] Salida solicitada por el usuario")
        break

# ===============================
# RESUMEN FINAL
# ===============================

print("\n" + "=" * 50)
print(f"  Frames procesados  : {frame_count}")
print(f"  Vehículos únicos   : {len(index)}")
print(f"  Frames guardados   : {total_saves}")
print(f"  Salida en          : {OUTPUT_DIR}")
print("=" * 50)

cap.release()
cv2.destroyAllWindows()
print("[INFO] Programa terminado")
