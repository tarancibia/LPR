"""
HERRAMIENTA: Definir ROI (Region of Interest)
==============================================
Abre el stream de la cámara en vivo y permite dibujar
un polígono con el mouse para definir la zona de interés.

Controles:
  Click izquierdo  → Agregar punto al polígono
  Click derecho    → Deshacer último punto
  S                → Guardar ROI en config/roi.json
  R                → Resetear polígono
  Q                → Salir sin guardar
"""

import cv2
import json
import numpy as np
import time
from pathlib import Path

# ===============================
# CONFIGURACION
# ===============================

RTSP_URL = "rtsp://admin:Muni2026@192.168.107.121:554/cam/realmonitor?channel=1&subtype=0"
CONFIG_PATH = Path("config/roi.json")

# Colores (BGR)
COLOR_POINT      = (0, 255, 255)    # Amarillo — puntos del polígono
COLOR_LINE       = (0, 200, 255)    # Naranja   — líneas del polígono
COLOR_FILL       = (0, 255, 0)      # Verde     — relleno semitransparente
COLOR_CLOSE_LINE = (0, 100, 255)    # Rojo      — línea de cierre
COLOR_TEXT       = (255, 255, 255)  # Blanco    — textos
COLOR_BG         = (30, 30, 30)     # Gris oscuro — fondo de UI

POINT_RADIUS = 6
LINE_THICKNESS = 2
FILL_ALPHA = 0.15  # Transparencia del relleno (0=invisible, 1=sólido)

# ===============================
# ESTADO GLOBAL
# ===============================

points = []          # Lista de puntos [(x, y), ...]
mouse_pos = (0, 0)   # Posición actual del mouse
frame_frozen = None  # Frame de referencia (se puede congelar)
frame_size = None    # (width, height)


def mouse_callback(event, x, y, flags, param):
    global points, mouse_pos

    mouse_pos = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"  + Punto {len(points):02d}: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            removed = points.pop()
            print(f"  - Punto eliminado: {removed}")


def draw_overlay(base_frame):
    """Dibuja el polígono y la UI sobre el frame."""
    frame = base_frame.copy()
    overlay = frame.copy()

    # ── Relleno semitransparente si hay 3+ puntos ──────────────────
    if len(points) >= 3:
        pts_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(overlay, [pts_array], COLOR_FILL)
        cv2.addWeighted(overlay, FILL_ALPHA, frame, 1 - FILL_ALPHA, 0, frame)

    # ── Líneas del polígono ────────────────────────────────────────
    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], COLOR_LINE, LINE_THICKNESS)

    # Línea de cierre (del último punto al primero)
    if len(points) >= 3:
        cv2.line(frame, points[-1], points[0], COLOR_CLOSE_LINE, LINE_THICKNESS, cv2.LINE_AA)

    # Línea fantasma: del último punto al mouse
    if points:
        cv2.line(frame, points[-1], mouse_pos, COLOR_LINE, 1, cv2.LINE_AA)

    # ── Puntos ────────────────────────────────────────────────────
    for i, (px, py) in enumerate(points):
        cv2.circle(frame, (px, py), POINT_RADIUS, COLOR_POINT, -1)
        cv2.circle(frame, (px, py), POINT_RADIUS + 2, (0, 0, 0), 1)
        # Número del punto
        cv2.putText(frame, str(i + 1), (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_POINT, 1, cv2.LINE_AA)

    # ── Panel de instrucciones (esquina superior izquierda) ────────
    lines_ui = [
        "DEFINIR ROI",
        "",
        "Click izq  → agregar punto",
        "Click der  → deshacer punto",
        "S          → guardar ROI",
        "R          → resetear",
        "Q          → salir sin guardar",
        "",
        f"Puntos: {len(points)}",
        f"Mouse: {mouse_pos}",
    ]

    pad = 10
    line_h = 18
    panel_w = 230
    panel_h = len(lines_ui) * line_h + pad * 2

    cv2.rectangle(frame, (pad, pad), (pad + panel_w, pad + panel_h), COLOR_BG, -1)
    cv2.rectangle(frame, (pad, pad), (pad + panel_w, pad + panel_h), (80, 80, 80), 1)

    for i, text in enumerate(lines_ui):
        color = (0, 220, 255) if i == 0 else COLOR_TEXT
        thickness = 2 if i == 0 else 1
        y_pos = pad * 2 + i * line_h
        cv2.putText(frame, text, (pad * 2, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, thickness, cv2.LINE_AA)

    # ── Estado en la parte inferior ────────────────────────────────
    h = frame.shape[0]
    if len(points) < 3:
        status = f"  Necesitas al menos 3 puntos  ({len(points)}/3)"
        s_color = (0, 100, 255)
    else:
        status = f"  ROI listo — {len(points)} puntos  |  Presiona S para guardar"
        s_color = (0, 200, 0)

    cv2.rectangle(frame, (0, h - 30), (frame.shape[1], h), COLOR_BG, -1)
    cv2.putText(frame, status, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, s_color, 1, cv2.LINE_AA)

    return frame


def save_roi():
    """Guarda el polígono actual en config/roi.json."""
    if len(points) < 3:
        print("[ERROR] Se necesitan al menos 3 puntos para guardar el ROI")
        return False

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "roi": [list(p) for p in points],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frame_size": {
            "width": frame_size[0],
            "height": frame_size[1],
        },
    }

    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n[OK] ROI guardado en '{CONFIG_PATH}'")
    print(f"     Puntos: {points}")
    return True


# ===============================
# CONEXION A LA CAMARA
# ===============================

print("[INFO] Conectando a la cámara...")
cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("[ERROR] No se pudo conectar a la cámara RTSP")
    exit(1)

print("[INFO] Cámara conectada")

# Leer primer frame para obtener resolución
ret, first_frame = cap.read()
if not ret:
    print("[ERROR] No se pudo leer el frame inicial")
    cap.release()
    exit(1)

frame_size = (first_frame.shape[1], first_frame.shape[0])
print(f"[INFO] Resolución: {frame_size[0]}x{frame_size[1]}")

# Cargar ROI existente si hay uno guardado
if CONFIG_PATH.exists():
    with open(CONFIG_PATH) as f:
        existing = json.load(f)
    points = [tuple(p) for p in existing.get("roi", [])]
    print(f"[INFO] ROI existente cargado: {len(points)} puntos")

# ===============================
# VENTANA Y MOUSE
# ===============================

WINDOW_NAME = "Definir ROI — LPR"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

print("\n[INFO] Ventana abierta — dibujá tu zona de interés con el mouse")
print("       Click izquierdo = agregar punto | Click derecho = deshacer")

# ===============================
# LOOP PRINCIPAL
# ===============================

while True:
    ret, frame = cap.read()
    if not ret:
        frame = first_frame.copy()  # Usar primer frame si hay error

    display = draw_overlay(frame)
    cv2.imshow(WINDOW_NAME, display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        print("[INFO] Saliendo sin guardar")
        break

    elif key == ord("s"):
        if save_roi():
            print("[INFO] Guardado. Podés cerrar o seguir ajustando.")

    elif key == ord("r"):
        points.clear()
        print("[INFO] Polígono reseteado")

cap.release()
cv2.destroyAllWindows()
print("[INFO] Herramienta cerrada")
