"""
LPR - Etapa 2: Extracción de Patentes
======================================
Este script implementa el Paso 1 y Paso 2 de la Etapa 2.
1. Lee los recortes de vehículos generados en la Etapa 1.
2. Selecciona los N frames más nítidos por vehículo.
3. Usa un modelo YOLO para detectar la patente dentro de esos recortes.
4. Guarda los recortes de la patente en una nueva estructura de carpetas.
"""

import cv2
import json
from pathlib import Path
from ultralytics import YOLO
import os

# ===============================
# CONFIGURACION
# ===============================

INPUT_DIR = Path("output_stage1/vehicles")
OUTPUT_DIR = Path("output_stage2/plates")

# Parámetros de Selección (Paso 1)
TOP_N_FRAMES = 3  # Cuántos frames nítidos analizar por vehículo

# Parámetros de Detección (Paso 2)
# IMPORTANTE: Reemplazar con un modelo entrenado para patentes.
# yolov8n.pt base no detectará patentes (no están en COCO).
PLATE_MODEL_PATH = "license_plate_detector.pt" 
PLATE_CONFIDENCE_THRESHOLD = 0.5

# ===============================
# PREPARAR DIRECTORIOS
# ===============================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# CARGAR MODELO
# ===============================
print(f"[INFO] Cargando modelo de patentes: {PLATE_MODEL_PATH}...")
try:
    model = YOLO(PLATE_MODEL_PATH)
except Exception as e:
    print(f"[ERROR] No se pudo cargar el modelo: {e}")
    exit(1)

# ===============================
# PROCESAMIENTO
# ===============================

def procesar_vehiculos():
    if not INPUT_DIR.exists():
        print(f"[ERROR] El directorio de entrada {INPUT_DIR} no existe.")
        return

    # Buscar todas las carpetas de tracks
    track_dirs = [d for d in INPUT_DIR.iterdir() if d.is_dir() and d.name.startswith("track_")]
    print(f"[INFO] Encontrados {len(track_dirs)} vehículos para analizar.")

    total_patentes_encontradas = 0

    for track_dir in track_dirs:
        track_id_str = track_dir.name
        meta_file = track_dir / "metadata.json"

        if not meta_file.exists():
            continue

        # --- PASO 1: SELECCION INTELIGENTE ---
        with open(meta_file, "r") as f:
            try:
                meta = json.load(f)
            except json.JSONDecodeError:
                print(f"[WARN] Error leyendo JSON: {meta_file}")
                continue

        frames = meta.get("frames", [])
        if not frames:
            continue

        # Ordenar por nitidez (sharpness) de mayor a menor
        frames_ordenados = sorted(frames, key=lambda x: x.get("sharpness", 0), reverse=True)
        mejores_frames = frames_ordenados[:TOP_N_FRAMES]

        # --- PASO 2: DETECCION Y CROP DE PATENTE ---
        # Crear carpeta de salida para este track
        out_track_dir = OUTPUT_DIR / track_id_str
        out_track_dir.mkdir(exist_ok=True)

        for frame_info in mejores_frames:
            img_filename = frame_info.get("file")
            img_path = track_dir / img_filename

            if not img_path.exists():
                continue

            # Leer la imagen del auto
            vehicle_img = cv2.imread(str(img_path))
            if vehicle_img is None:
                continue

            # Inferencia para buscar la patente
            results = model(vehicle_img, conf=PLATE_CONFIDENCE_THRESHOLD, verbose=False)
            boxes = results[0].boxes

            # Si encuentra alguna patente
            if len(boxes) > 0:
                # Tomar la de mayor confianza (suele ser la primera o iteramos para buscar el max)
                best_box = boxes[0] # Asumimos 1 sola patente visible
                x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
                
                # Recortar la patente
                plate_crop = vehicle_img[y1:y2, x1:x2]

                if plate_crop.size > 0:
                    # Guardar el recorte
                    plate_filename = f"plate_{img_filename}"
                    cv2.imwrite(str(out_track_dir / plate_filename), plate_crop)
                    total_patentes_encontradas += 1
                    print(f"[{track_id_str}] Patente guardada: {plate_filename}")
            else:
                print(f"[{track_id_str}] No se detectó patente en {img_filename}")

    print("\n" + "="*50)
    print(f"[INFO] Procesamiento de la Etapa 2 finalizado.")
    print(f"[INFO] Patentes recortadas y guardadas: {total_patentes_encontradas}")
    print("="*50)

if __name__ == "__main__":
    procesar_vehiculos()
