"""
LPR - Etapa 1: Detección de Vehículos
=======================================
Detecta vehículos en el stream RTSP y guarda:
  - Coordenadas del bounding box (JSON)
  - Crop del vehículo en resolución original (JPG)

Las coordenadas y crops quedan listos para la Etapa 2 (detección de patente).

ROI: Si existe config/roi.json, solo se procesan vehículos dentro de esa zona.
     Usar tools/define_roi.py para definir/editar el polígono.
"""

import cv2
import json
import os
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ===============================
# CONFIGURACION
# ===============================

# Stream de la cámara Dahua
RTSP_URL = "rtsp://admin:gda.adm123@192.168.107.121:554/cam/realmonitor?channel=1&subtype=0"

# Modelo YOLO (detector genérico COCO)
MODEL_PATH = "yolov8n.pt"

# Procesar 1 de cada N frames para ahorrar CPU
PROCESS_EVERY = 3

# Confianza mínima para aceptar una detección de vehículo (0.0 a 1.0)
CONFIDENCE_THRESHOLD = 0.5

# Clases de vehículos en el dataset COCO
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Carpeta raíz donde se guardan los resultados
OUTPUT_DIR  = Path("output_stage1")
ROI_CONFIG  = Path("config/roi.json")

# Máximo de vehículos a guardar en disco (None = sin límite)
MAX_SAVES = None

# ===============================
# PREPARAR DIRECTORIOS
# ===============================

crops_dir = OUTPUT_DIR / "crops"
crops_dir.mkdir(parents=True, exist_ok=True)

coords_file = OUTPUT_DIR / "coordenadas.json"

# Cargar coordenadas previas si ya existe el archivo
if coords_file.exists():
    with open(coords_file, "r") as f:
        all_detections = json.load(f)
    print(f"[INFO] {len(all_detections)} detecciones previas cargadas desde {coords_file}")
else:
    all_detections = []

# ===============================
# CARGAR ROI (si existe)
# ===============================

roi_polygon = None  # None = sin filtro, detecta en todo el frame

if ROI_CONFIG.exists():
    with open(ROI_CONFIG) as f:
        roi_data = json.load(f)
    roi_polygon = np.array(roi_data["roi"], dtype=np.int32)
    print(f"[INFO] ROI cargado: {len(roi_data['roi'])} puntos — solo se procesarán vehículos dentro de la zona")
else:
    print("[WARN] No se encontró config/roi.json — se detecta en todo el frame")
    print("       Ejecutar 'python tools/define_roi.py' para definir la zona de interés")


def center_in_roi(x1, y1, x2, y2):
    """Retorna True si el centro del bbox está dentro del polígono ROI."""
    if roi_polygon is None:
        return True  # Sin ROI: aceptar todo
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    # pointPolygonTest: >0 adentro, 0 borde, <0 afuera
    return cv2.pointPolygonTest(roi_polygon, (cx, cy), False) >= 0

# ===============================
# CARGA DEL MODELO
# ===============================

print("[INFO] Cargando modelo YOLO...")
model = YOLO(MODEL_PATH)
print(f"[INFO] Modelo '{MODEL_PATH}' listo")

# ===============================
# CONEXION A LA CAMARA
# ===============================

print(f"[INFO] Conectando a: {RTSP_URL.split('@')[1].split('/')[0]}")
cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("[ERROR] No se pudo conectar a la cámara RTSP")
    print("        Verificar IP, usuario y contraseña")
    exit(1)

print("[INFO] Cámara conectada")

# ===============================
# VARIABLES DE ESTADO
# ===============================

frame_count = 0
saves_count = len(all_detections)
prev_time = time.time()

# ===============================
# LOOP PRINCIPAL
# ===============================

print("\n[INFO] Iniciando detección de vehículos...")
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

    # Saltear frames para reducir carga de CPU
    if frame_count % PROCESS_EVERY != 0:
        continue

    # Timestamp del frame procesado
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # --------------------------------------------------
    # INFERENCIA: detectar objetos en el frame completo
    # --------------------------------------------------
    results = model(frame, verbose=False)
    boxes = results[0].boxes

    # Frame de anotación (no modifica el original)
    annotated = frame.copy()
    vehicles_this_frame = []

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        # Filtrar solo vehículos con suficiente confianza
        if cls_id not in VEHICLE_CLASSES or conf < CONFIDENCE_THRESHOLD:
            continue

        # Coordenadas en píxeles del frame original (alta resolución)
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        class_name = VEHICLE_CLASSES[cls_id]

        # Filtrar por ROI: ignorar vehículos fuera de la zona de interés
        if not center_in_roi(x1, y1, x2, y2):
            continue

        vehicles_this_frame.append({
            "class_id": cls_id,
            "class_name": class_name,
            "confidence": round(conf, 4),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })

        # --------------------------------------------------
        # DIBUJAR bounding box en pantalla
        # --------------------------------------------------
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"{class_name} {conf:.0%}"
        cv2.putText(
            annotated, label, (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2
        )

        # --------------------------------------------------
        # GUARDAR coordenadas + crop si no se llegó al límite
        # --------------------------------------------------
        if MAX_SAVES is None or saves_count < MAX_SAVES:

            # Crop del vehículo en resolución ORIGINAL (sin escalar)
            crop = frame[y1:y2, x1:x2]

            if crop.size > 0:
                crop_filename = f"{timestamp}_f{frame_count}_{class_name}_{saves_count + 1:04d}.jpg"
                crop_path = crops_dir / crop_filename
                cv2.imwrite(str(crop_path), crop)

                # Registro de la detección
                detection_record = {
                    "id": saves_count + 1,
                    "timestamp": timestamp,
                    "frame": frame_count,
                    "crop_file": crop_filename,
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": round(conf, 4),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "frame_size": {
                        "width": frame.shape[1],
                        "height": frame.shape[0],
                    },
                }
                all_detections.append(detection_record)
                saves_count += 1

                print(
                    f"[SAVE] #{saves_count:04d} | frame={frame_count} | "
                    f"{class_name} ({conf:.0%}) | "
                    f"bbox=({x1},{y1})-({x2},{y2}) | "
                    f"crop={crop_filename}"
                )

                # Guardar JSON actualizado en disco
                with open(coords_file, "w") as f:
                    json.dump(all_detections, f, indent=2)

    # --------------------------------------------------
    # INFO EN PANTALLA
    # --------------------------------------------------
    current_time = time.time()
    fps = 1.0 / max(current_time - prev_time, 1e-6)
    prev_time = current_time

    # Dibujar el polígono ROI sobre el frame
    if roi_polygon is not None:
        cv2.polylines(annotated, [roi_polygon], isClosed=True, color=(0, 255, 255), thickness=2)

    cv2.putText(annotated, f"FPS: {fps:.1f}",          (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(annotated, f"Frame: {frame_count}",     (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(annotated, f"Vehiculos: {len(vehicles_this_frame)}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
    cv2.putText(annotated, f"Guardados: {saves_count}",  (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    roi_status = "ROI: ACTIVO" if roi_polygon is not None else "ROI: sin definir"
    roi_color  = (0, 255, 255) if roi_polygon is not None else (0, 100, 255)
    cv2.putText(annotated, roi_status, (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 1)

    if MAX_SAVES and saves_count >= MAX_SAVES:
        cv2.putText(
            annotated, "LIMITE ALCANZADO — Solo mostrando",
            (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1
        )

    cv2.imshow("LPR - Etapa 1: Deteccion de Vehiculos", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\n[INFO] Salida solicitada por el usuario")
        break

# ===============================
# LIMPIEZA Y RESUMEN
# ===============================

print("\n" + "=" * 50)
print(f"  Frames procesados : {frame_count // PROCESS_EVERY}")
print(f"  Vehículos guardados: {saves_count}")
print(f"  Coordenadas en    : {coords_file}")
print(f"  Crops en          : {crops_dir}")
print("=" * 50)

cap.release()
cv2.destroyAllWindows()
print("[INFO] Programa terminado")
