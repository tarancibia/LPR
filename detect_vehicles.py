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
# CONFIGURACION
# ===============================

RTSP_URL   = "rtsp://admin:gda.adm123@192.168.107.121:554/cam/realmonitor?channel=1&subtype=0"
MODEL_PATH = "yolov8n.pt"

# Procesar 1 de cada N frames
PROCESS_EVERY = 3

# Confianza mínima para aceptar una detección
CONFIDENCE_THRESHOLD = 0.5

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


def center_in_roi(x1, y1, x2, y2):
    """Retorna True si el centro del bbox está dentro del polígono ROI."""
    if roi_polygon is None:
        return True
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cv2.pointPolygonTest(roi_polygon, (cx, cy), False) >= 0


def track_dir(track_id: int) -> Path:
    """Retorna la carpeta del vehículo, creándola si no existe."""
    d = vehicles_dir / f"track_{track_id:04d}"
    d.mkdir(exist_ok=True)
    return d


def save_frame(track_id: int, frame_number: int, crop, bbox, conf, class_name, timestamp):
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
    # INFERENCIA CON TRACKING (ByteTrack)
    # --------------------------------------------------
    results = model.track(
        frame,
        persist=True,           # mantiene IDs entre frames
        tracker="bytetrack.yaml",
        verbose=False,
        conf=CONFIDENCE_THRESHOLD,
        classes=list(VEHICLE_CLASSES.keys()),
    )

    boxes    = results[0].boxes
    track_ids = boxes.id

    # Sin detecciones con ID asignado en este frame
    if track_ids is None:
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

    track_ids_list = track_ids.int().cpu().tolist()

    for box, tid_raw in zip(boxes, track_ids_list):
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])

        if cls_id not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        class_name = VEHICLE_CLASSES[cls_id]

        # Filtrar por ROI
        if not center_in_roi(x1, y1, x2, y2):
            continue

        # Asignar ID secuencial solo a los que entran a la ROI
        if tid_raw not in track_mapping:
            track_mapping[tid_raw] = next_track_id
            next_track_id += 1
        tid = track_mapping[tid_raw]

        # Crop en resolución original
        crop = frame[y1:y2, x1:x2]

        if crop.size > 0:
            fname = save_frame(tid, frame_count, crop, (x1, y1, x2, y2), conf, class_name, timestamp)
            total_saves += 1
            print(
                f"[SAVE] track={tid:04d} | frame={frame_count} | "
                f"{class_name} ({conf:.0%}) | "
                f"bbox=({x1},{y1})-({x2},{y2}) | {fname}"
            )

        # Dibujar bbox con ID
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"ID:{tid} {class_name} {conf:.0%}"
        cv2.putText(annotated, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)

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
print(f"  Frames procesados  : {frame_count // PROCESS_EVERY}")
print(f"  Vehículos únicos   : {len(index)}")
print(f"  Frames guardados   : {total_saves}")
print(f"  Salida en          : {OUTPUT_DIR}")
print("=" * 50)

cap.release()
cv2.destroyAllWindows()
print("[INFO] Programa terminado")
