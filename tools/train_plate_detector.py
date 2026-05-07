from ultralytics import YOLO
import torch

def train_plate_detector():
    # 1. Cargar el modelo base más ligero y rápido
    print("[INFO] Cargando modelo base YOLOv8 nano...")
    model = YOLO("yolov8n.pt")
    
    # IMPORTANTE: Cambia esta ruta para que apunte al archivo data.yaml 
    # que descargaste de Roboflow.
    DATASET_YAML = "datasets/license_plates/data.yaml"

    print(f"[INFO] Iniciando entrenamiento usando el dataset: {DATASET_YAML}")
    print("[INFO] Esto puede tardar dependiendo de tu procesador o tarjeta gráfica...")

    # 2. Iniciar el entrenamiento
    results = model.train(
        data=DATASET_YAML,
        epochs=30,          # 30 pasadas suele ser suficiente para patentes
        imgsz=640,          # Resolución estándar
        batch=16,           # Autos a procesar a la vez (bájalo a 8 si te da error de memoria)
        name="detector_patentes", # Nombre de la carpeta de resultados
        device="0" if torch.cuda.is_available() else "cpu" # Usa GPU si tienes, si no CPU
    )

    print("\n[ÉXITO] Entrenamiento finalizado.")
    print("Tu nuevo modelo entrenado está guardado en:")
    print("runs/detect/detector_patentes/weights/best.pt")
    print("¡Copia ese archivo best.pt y renómbralo a license_plate_detector.pt para usarlo en la Etapa 2!")

if __name__ == "__main__":
    train_plate_detector()
