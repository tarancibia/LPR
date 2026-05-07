# Puntos críticos (riesgos / bugs / maintainability)

Este proyecto implementa la **Etapa 1**: detección y tracking de vehículos desde RTSP y persistencia de crops + metadata en `output_stage1/`.

## Seguridad / configuración sensible

- **Credenciales RTSP hardcodeadas**: `RTSP_URL` contiene usuario/clave en texto plano en:
  - `detect_vehicles.py`
  - `tools/define_roi.py`
- **Riesgo**: exposición accidental en git/logs/screenshots y acoplamiento a una cámara/IP específica.
- **Acción sugerida**:
  - mover `RTSP_URL` a variables de entorno o a un archivo `config.json` fuera del control de versiones
  - agregar `config/*.local.*` al `.gitignore` si aplica

## Artefactos generados versionados (ruido en git)

- **No hay `.gitignore`** y por eso aparecen cambios masivos en `output_stage1/` (JPGs + JSONs) y potencialmente `venv/`.
- **Riesgo**: historial pesado, diffs inútiles, conflictos frecuentes, repo difícil de clonar/compartir.
- **Acción sugerida**:
  - ignorar `output_stage1/` completo (o al menos `output_stage1/vehicles/**`)
  - ignorar `venv/`
  - si querés reproducibilidad, versionar solo `config/roi.json` (opcional)

## Performance / I/O en disco (escala mal)

- En cada guardado (`save_frame`) se hace:
  - `cv2.imwrite(...)` del JPG
  - lectura + escritura completa de `metadata.json` (crece sin límite)
  - escritura completa de `index.json`
- **Riesgo**: el costo de JSON se vuelve alto con tracks largos (tiempo y desgaste de disco), y puede haber corrupción si el proceso se corta en medio de escritura.
- **Acción sugerida**:
  - acumular en memoria y hacer flush por intervalos (cada N frames o cada X segundos)
  - usar escrituras atómicas (escribir a `*.tmp` y renombrar)
  - opcional: usar un formato tipo SQLite/Parquet para metadata

## Dependencias / reproducibilidad del entorno

- No hay `requirements.txt` / `pyproject.toml`.
- Imports clave: `opencv-python` (`cv2`), `ultralytics`, `numpy`.
- **Riesgo**: difícil reproducir entorno (versiones, CUDA, OpenCV).
- **Acción sugerida**:
  - agregar `requirements.txt` (y documentar versión de Python)

## Tracking: referencia a `bytetrack.yaml`

- `detect_vehicles.py` llama `model.track(... tracker="bytetrack.yaml" ...)` pero el repo no versiona ese archivo.
- **Riesgo**: si Ultralytics no lo resuelve como recurso builtin, falla en runtime.
- **Acción sugerida**:
  - confirmar que Ultralytics lo trae por defecto; si no, incluir el YAML o usar ruta explícita

## Portabilidad de paths (Windows vs Linux)

- `index.json` guarda `folder` como string con separadores de Windows (ej: `"vehicles\\track_0001"`).
- **Riesgo**: una “Etapa 2” en Linux puede fallar si espera `/`.
- **Acción sugerida**:
  - guardar rutas normalizadas (por ejemplo, POSIX) o guardar `track_id` y construir rutas con `Path` en la etapa consumidora

## Validación de ROI vs resolución actual

- `config/roi.json` guarda `frame_size`, pero `detect_vehicles.py` no valida que coincida con la resolución actual del stream.
- **Riesgo**: ROI inválido si cambia la cámara/resolución/stream.
- **Acción sugerida**:
  - validar `frame_size` y avisar/rehusar si hay mismatch

## Ejecución no-headless / UI obligatoria

- Se usa `cv2.imshow`/`cv2.waitKey` en ambos scripts.
- **Riesgo**: en servidores/headless falla o requiere workarounds.
- **Acción sugerida**:
  - flag “headless” para desactivar UI y loggear métricas por consola

## Estructura del script (testeo/importe)

- El script ejecuta lógica y crea directorios al importarse (no hay `main()`/`if __name__ == "__main__":`).
- **Riesgo**: dificulta testear, reutilizar como módulo o integrar en otro pipeline.
- **Acción sugerida**:
  - encapsular en funciones y agregar `main()`

## Procesamiento de Imágenes: Nitidez y Foco

- **Sensibilidad al ruido en `calculate_sharpness`**: El Laplaciano es muy sensible al ruido digital (especialmente en condiciones de baja luz).
- **Riesgo**: El sistema podría elegir una imagen borrosa pero con mucho "grano" (ruido) pensando que es nítida.
- **Acción sugerida**: 
  - Aplicar un `cv2.GaussianBlur(gray, (3, 3), 0)` antes del Laplaciano para filtrar el ruido de alta frecuencia y obtener una medida de nitidez más robusta.
- **Ajuste de Padding en `expand_bbox_for_crop`**: Los ratios de expansión (`BBOX_PADDING_RATIO_X`, `Y_TOP`, `Y_BOTTOM`) son fijos.
- **Riesgo**: Si el padding es muy pequeño, se puede cortar la patente. Si es muy grande, se guarda demasiada imagen innecesaria (más disco y más ruido para el OCR).
- **Acción sugerida**: 
  - Revisar si los márgenes actuales son óptimos para las cámaras instaladas y si el padding asimétrico está cubriendo bien las patentes bajas.
