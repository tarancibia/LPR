"""
tools/download_dataset.py
==========================
Script para descargar y preparar el dataset de patentes desde Roboflow.

CUÁNDO USAR ESTE SCRIPT:
    Ejecutarlo la primera vez que se clona el proyecto, o si la carpeta
    'datasets/' fue borrada (está excluida del repositorio por su tamaño).

DATASET:
    Fuente: Roboflow Universe — License Plate Recognition
    URL:    https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4
    Formato: YOLOv8
    Clases: 1 (License_Plate)

USO:
    (venv) $ python tools/download_dataset.py

RESULTADO:
    Crea la carpeta 'datasets/license_plates/' con la siguiente estructura:
        datasets/license_plates/
            data.yaml       <- archivo de configuración para el entrenamiento
            train/          <- imágenes y etiquetas de entrenamiento (~80%)
            valid/          <- imágenes y etiquetas de validación (~10%)
            test/           <- imágenes y etiquetas de prueba (~10%)

NOTA: Si el link de descarga directa caducó, entrar a Roboflow y generar
      uno nuevo desde el botón "Download Dataset" (formato YOLOv8).
"""

import urllib.request
import zipfile
import os

url = 'https://universe.roboflow.com/ds/07x7T9oNMn?key=nBYVZwV26y'
zip_path = 'dataset.zip'
extract_dir = 'datasets/license_plates'

print('[INFO] Descargando el dataset desde Roboflow (esto puede tardar un minuto)...')
try:
    urllib.request.urlretrieve(url, zip_path)
    print(f'[INFO] Archivo descargado exitosamente: {zip_path}')
except Exception as e:
    print(f'[ERROR] Falló la descarga: {e}')
    print('[HINT] El link de Roboflow puede haber caducado.')
    print('       Ingresá a https://universe.roboflow.com y generá un nuevo link de descarga.')
    exit(1)

print(f'[INFO] Extrayendo los archivos en {extract_dir}...')
try:
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print('[INFO] Extracción completada.')

    # Corregir las rutas del data.yaml (Roboflow usa rutas relativas con '../')
    yaml_path = os.path.join(extract_dir, 'data.yaml')
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            content = f.read()
        content = content.replace('../train/', 'train/').replace('../valid/', 'valid/').replace('../test/', 'test/')
        with open(yaml_path, 'w') as f:
            f.write(content)
        print('[INFO] Rutas en data.yaml corregidas automáticamente.')

    os.remove(zip_path)
    print('[INFO] Archivo .zip temporal eliminado.')
except Exception as e:
    print(f'[ERROR] Falló la extracción: {e}')
    exit(1)

print('\n[ÉXITO] El dataset está listo.')
print('        Ya podés ejecutar: python tools/train_plate_detector.py')
