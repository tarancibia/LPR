# Planificación Etapa 2: Detección y Lectura de Patentes (LPR)

Este documento detalla el pipeline propuesto para tomar los recortes de vehículos generados en la **Etapa 1** (`output_stage1/vehicles/`) y extraer el texto consolidado de las patentes.

---

## Flujo de Trabajo (Pipeline)

### Paso 1: Selección Inteligente de Frames (Pre-filtro)
Dado que un mismo `track_id` puede tener docenas de fotos, no es eficiente ni necesario analizar todas.
1. **Lectura de Metadatos:** El script leerá el archivo `metadata.json` de cada carpeta de vehículo.
2. **Ranking por Nitidez:** Ordenará los frames guardados basándose en el valor de `sharpness` (calculado en la Etapa 1).
3. **Selección:** Seleccionará únicamente los **Top 3 o Top 5 frames más nítidos** para enviarlos a la siguiente fase.

### Paso 2: Detección de la Chapa Patente (Plate Detection)
Una vez seleccionados los mejores recortes del auto:
1. **Inferencia Local:** Se pasará el recorte del auto por un modelo ligero especializado en detectar matrículas (por ejemplo, un modelo YOLO nano `license_plate_detector.pt`).
2. **Extracción (Crop) y Guardado:** Se obtendrán las coordenadas (bounding box) de la patente dentro del auto, se realizará el recorte final y **se guardará en una nueva estructura de carpetas** (ej. `output_stage2/plates/track_0001/plate_frame_0042.jpg`).
3. *Nota:* Hacer esto sobre el recorte del auto y no sobre el frame completo reduce drásticamente los falsos positivos (ej. carteles en la calle).

### Paso 3: Pre-procesamiento de la Imagen (Mejora para OCR)
Los motores de OCR necesitan imágenes claras y con buen contraste. A los recortes de la patente se les aplicará:
1. **Conversión a Grises:** Simplifica el mapa de bits.
2. **Mejora de Contraste (CLAHE):** Ayuda a lidiar con reflejos del sol o sombras duras en la patente.
3. **Escalado (Upscaling):** Si la patente es muy pequeña (menos de 60px de alto), se escala usando interpolación cúbica.
4. **(Opcional) Corrección de Perspectiva:** Si el bounding box detecta una inclinación severa.

### Paso 4: Lectura (OCR) y Consolidación
1. **Lectura Óptica:** Pasar la imagen procesada por el motor de OCR (ej. PaddleOCR o EasyOCR).
2. **Filtrado Regex:** Limpiar la salida usando expresiones regulares para forzar formatos válidos (ej. Mercosur: `[A-Z]{2}[0-9]{3}[A-Z]{2}`).
3. **Consolidación (Sistema de Votación):**
   * Como analizamos (por ejemplo) 3 frames del mismo auto, obtendremos 3 lecturas.
   * Si las lecturas son `["AB123CD", "AB123CD", "AB128CD"]`, el sistema elige `"AB123CD"` por mayoría (o basándose en el nivel de confianza arrojado por el OCR).

### Paso 5: Persistencia de Resultados
1. Crear un registro final unificado en un archivo (ej. `output_stage2/results.json` o base de datos SQLite).
2. Guardar el mejor crop de la patente propiamente dicha para propósitos de evidencia o auditoría visual.

---

## Decisiones Técnicas Pendientes

Para comenzar con la implementación, necesitamos definir:

1. **Modelo Detector de Patentes (Paso 2):**
   * **Seleccionado:** `license_plate_detector.pt` (Entrenado a partir de YOLOv8n).
2. **Motor OCR (Paso 4):**
   * **Seleccionado:** **EasyOCR** (Restaurado por compatibilidad con Python 3.14).
   * PaddleOCR: Descartado temporalmente por falta de compatibilidad con Python 3.14.
