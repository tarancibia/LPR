import os
import glob
import cv2
import numpy as np
import easyocr

def test_ocr_restored():
    print("[INFO] Cargando modelo EasyOCR (Restaurado)...")
    # Usamos inglés para patentes
    reader = easyocr.Reader(['en'], gpu=False)
    
    # Obtener lista de carpetas de tracks
    track_folders = glob.glob(os.path.join("output_stage2", "plates", "track_*"))
    
    if not track_folders:
        print("[ERROR] No se encontraron carpetas de tracks en output_stage2/plates/")
        return
        
    print(f"[INFO] Encontrados {len(track_folders)} vehículos para analizar.\n")

    for folder in track_folders:
        plate_images = glob.glob(os.path.join(folder, "*.jpg"))
        if not plate_images: continue
        
        # Tomamos la primera imagen de cada track para la prueba
        test_image_path = plate_images[0]
        track_name = os.path.basename(folder)
        
        print(f"\n[INFO] Analizando {track_name}: {os.path.basename(test_image_path)}")
        
        # Leer imagen
        img = cv2.imread(test_image_path)
        if img is None: continue

        # --- PRE-PROCESAMIENTO ---
        # 1. Agrandar la imagen x3 (Crucial para recortes pequeños)
        h, w = img.shape[:2]
        img_scaled = cv2.resize(img, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
        
        # 2. Nitidez
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        processed = cv2.filter2D(img_scaled, -1, kernel)

        # --- EJECUTAR EASYOCR ---
        # Usamos allowlist para evitar símbolos raros
        results = reader.readtext(
            processed, 
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )

        print("-" * 35)
        if len(results) == 0:
            print("-> RESULTADO: No se detectó texto.")
        else:
            for (bbox, text, prob) in results:
                clean_text = text.replace(" ", "").upper()
                print(f"-> RESULTADO: '{clean_text}' | Confianza: {prob*100:.2f}%")
        print("-" * 35)

if __name__ == "__main__":
    test_ocr_restored()
