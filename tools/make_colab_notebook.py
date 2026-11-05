#!/usr/bin/env python3
"""Genera el cuaderno de Colab que entrena el detector de ratones (RF-DETR): ``colab/entrenar_rfdetr.ipynb``.

Uso (desde la raíz del proyecto):
    python tools/make_colab_notebook.py
    python tools/make_colab_notebook.py --salida otra/ruta.ipynb

El cuaderno se GENERA (no se edita a mano) para que la decodificación del ONNX sea siempre la de
``learning/onnx_decode.py``, la misma que usará TrackerLab: si cambia ese módulo, se vuelve a generar el cuaderno.
Las pruebas (tests/test_colab_notebook.py) comprueban que el cuaderno trae exactamente esa fuente y lo ejecutan completo
contra dobles de torch, rfdetr, onnxruntime y google.colab (no prueba que RF-DETR funcione, sí que este código de
pegamento no tiene errores).

Lo que este cuaderno NO hace: la evaluación honesta. La validación que calcula usa videos de validación con etiquetas
automáticas (o revisadas); la medición de verdad se hace después, en tu Mac, con el video reservado y tus etiquetas a mano.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECODE_PATH = os.path.join(ROOT, "learning", "onnx_decode.py")
DEFAULT_OUT = os.path.join(ROOT, "colab", "entrenar_rfdetr.ipynb")


def _lines(text: str) -> List[str]:
    text = text.strip("\n")
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def md(text: str) -> Dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> Dict[str, Any]:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _lines(text)}


def decode_source() -> str:
    """Fuente de learning/onnx_decode.py para incrustar en una celda (sin ``from __future__``, que solo vale al inicio de un módulo)."""
    with open(DECODE_PATH, encoding="utf-8") as fh:
        src = fh.read()
    linea = "from __future__ import annotations\n"
    if src.count(linea) != 1:
        raise RuntimeError("learning/onnx_decode.py debe tener exactamente un «from __future__ import annotations».")
    return src.replace(linea, "")


# ----------------------------------------------------------------------
# Celdas
# ----------------------------------------------------------------------
INTRO = """
# Entrenar el detector de ratones (RF-DETR)

Este cuaderno afina un detector preentrenado (**RF-DETR**, licencia Apache 2.0) con las imágenes recortadas al ROI que arma `tools/build_training_set.py`, lo evalúa y lo exporta a **ONNX** para usarlo dentro de TrackerLab.

**Antes de empezar**
1. **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU** (una T4 sirve). Sin GPU el entrenamiento es demasiado lento.
2. Ten a la mano `entrenamiento.zip`: la carpeta que dejó `build_training_set.py` (con `data.yaml`, `train/` y `valid/`), comprimida. Pesa decenas de MB; los videos nunca se suben.
3. Ejecuta las celdas **de arriba abajo**, una por una. Si Colab pide reiniciar el entorno después de instalar, hazlo y vuelve a ejecutar desde la celda de configuración.

**Qué esperar**
- Primero corre una **prueba de 2 épocas** para medir tiempos y confirmar que todo funciona. Sus números no significan nada.
- Después el entrenamiento real, que se detiene solo si deja de mejorar (parada temprana).
- Las cifras de validación que imprime este cuaderno salen de los videos de validación, con etiquetas automáticas o revisadas. **No son la evaluación final**: esa se hace en tu Mac con el video reservado y tus etiquetas dibujadas a mano.

**Tu espacio en Drive es limitado.** El entrenamiento escribe en el disco local de Colab y a Drive solo se copia lo necesario (el mejor modelo, el ONNX y los informes). Lo local se pierde cuando Colab se desconecta; lo copiado a Drive no.
"""

CONFIG = '''
# ============ CONFIGURACIÓN (lo único que normalmente se toca) ============
MODELO = "small"           # "nano" = más rápido; "small" = más preciso. Entrena uno y luego el otro con los mismos datos.
RESOLUCION = None          # None = la de fábrica del modelo. Si pones un número, debe ser cuadrado y válido (el entrenador avisa).
EPOCAS_PRUEBA = 2          # la corrida corta que mide tiempos
EPOCAS = 100               # tope del entrenamiento real (la parada temprana suele pararlo antes)
LOTE = 4                   # imágenes por paso...
ACUMULACION = 4            # ...y pasos acumulados: LOTE x ACUMULACION = 16, lo que recomienda la documentación
LR = None                  # None = el de fábrica (1e-4). Si el entrenamiento no mejora, prueba la mitad (5e-5)
PACIENCIA = 10             # épocas sin mejora antes de parar
EPOCAS_A_IGNORAR = 3       # no elegir «mejor punto» ni parar durante las primeras épocas
UMBRAL = 0.5               # confianza mínima para contar una detección
VERSION_RFDETR = None      # None = la última. Para REPRODUCIR un entrenamiento, escribe la versión que quedó anotada en run_info.json
GUARDAR_EN_DRIVE = True    # copia el resultado final a tu Drive (sobrevive si Colab se desconecta)
CARPETA_DRIVE = "TrackerLab_Entrenamiento"
ZIP_EN_DRIVE = None        # p. ej. "TrackerLab_Entrenamiento/entrenamiento.zip" (ruta dentro de Mi unidad); None = lo subes con un botón
RAIZ_DRIVE = "/content/drive/MyDrive"
RAIZ_LOCAL = "/content"
'''

INSTALL = '''
# ============ Paso 1 · Instalar RF-DETR ============
paquete = "rfdetr[train,onnx]" + (f"=={VERSION_RFDETR}" if VERSION_RFDETR else "")
!pip install -q "{paquete}" onnxruntime

import importlib.metadata as md
try:
    VERSION_INSTALADA = md.version("rfdetr")
except Exception:
    VERSION_INSTALADA = "desconocida"
print("rfdetr", VERSION_INSTALADA, "(anótala: con ella se puede reproducir este entrenamiento)")
'''

GPU = '''
# ============ Paso 2 · Comprobar la GPU ============
import torch

if not torch.cuda.is_available():
    raise RuntimeError("No hay GPU. Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU, y vuelve a empezar.")
GPU = torch.cuda.get_device_name(0)
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {GPU} · memoria: {VRAM_GB:.1f} GB")

# Valores que sugiere la documentación de RF-DETR para un lote efectivo de 16, según la memoria de la GPU
if VRAM_GB >= 38:
    SUGERIDO = (16, 1)
elif VRAM_GB >= 22:
    SUGERIDO = (8, 2)
elif VRAM_GB >= 15:
    SUGERIDO = (4, 4)
elif VRAM_GB >= 11:
    SUGERIDO = (2, 8)
else:
    SUGERIDO = (1, 16)
print(f"Lote sugerido para esta GPU: LOTE={SUGERIDO[0]}, ACUMULACION={SUGERIDO[1]}  (ahora: LOTE={LOTE}, ACUMULACION={ACUMULACION})")
if (LOTE, ACUMULACION) != SUGERIDO:
    print("→ Si ves un error de memoria, ajusta LOTE y ACUMULACION en la celda de configuración (su producto debe seguir siendo 16).")
'''

DRIVE = '''
# ============ Paso 3 · Carpetas de trabajo y Drive ============
import json, os, random, shutil, time, zipfile
from datetime import datetime

if GUARDAR_EN_DRIVE or ZIP_EN_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")

NOMBRE_CORRIDA = f"rfdetr_{MODELO}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
SALIDA = os.path.join(RAIZ_LOCAL, "salidas", NOMBRE_CORRIDA)      # disco local: aquí entrena (los checkpoints pesan)
ENTREGA = os.path.join(RAIZ_DRIVE if GUARDAR_EN_DRIVE else RAIZ_LOCAL, CARPETA_DRIVE if GUARDAR_EN_DRIVE else "entregas", NOMBRE_CORRIDA)
os.makedirs(SALIDA, exist_ok=True)
print("Entrena en:", SALIDA)
print("El resultado final se copiará a:", ENTREGA)
'''

DATOS = '''
# ============ Paso 4 · Traer el conjunto de datos ============
DATOS = os.path.join(RAIZ_LOCAL, "datos")
shutil.rmtree(DATOS, ignore_errors=True)
os.makedirs(DATOS)

if ZIP_EN_DRIVE:
    ruta_zip = os.path.join(RAIZ_DRIVE, ZIP_EN_DRIVE)
else:
    from google.colab import files
    subido = files.upload()          # elige entrenamiento.zip
    if not subido:
        raise RuntimeError("No se subió ningún archivo.")
    ruta_zip = os.path.join(os.getcwd(), next(iter(subido)))

with zipfile.ZipFile(ruta_zip) as z:
    z.extractall(DATOS)
raices = [d for d, _, fs in os.walk(DATOS) if "data.yaml" in fs]
if len(raices) != 1:
    raise RuntimeError(f"El zip debe contener exactamente una carpeta con data.yaml (encontré {len(raices)}).")
DATASET_DIR = raices[0]
print("Conjunto en:", DATASET_DIR)
'''

REVISAR = '''
# ============ Paso 5 · Revisar el conjunto ANTES de entrenar ============
import cv2
import numpy as np
import matplotlib.pyplot as plt

EXT = (".jpg", ".jpeg", ".png")


def listar(split):
    d = os.path.join(DATASET_DIR, split, "images")
    if not os.path.isdir(d):
        raise RuntimeError(f"Falta la carpeta {d}")
    return sorted(f for f in os.listdir(d) if f.lower().endswith(EXT))


def leer_caja(split, archivo):
    """(cx, cy, ancho, alto) normalizados de la etiqueta de una imagen. Lanza ValueError si algo no está bien."""
    ruta = os.path.join(DATASET_DIR, split, "labels", os.path.splitext(archivo)[0] + ".txt")
    if not os.path.exists(ruta):
        raise ValueError("no tiene archivo de etiqueta")
    with open(ruta, encoding="utf-8-sig") as fh:
        lineas = [ln for ln in fh.read().splitlines() if ln.strip()]
    if len(lineas) != 1:
        raise ValueError(f"tiene {len(lineas)} cajas (debe tener exactamente 1)")
    p = lineas[0].split()
    if len(p) != 5 or p[0] != "0":
        raise ValueError("formato inválido (se espera: 0 cx cy ancho alto)")
    cx, cy, w, h = (float(v) for v in p[1:])
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
        raise ValueError("valores fuera de rango")
    return cx, cy, w, h


CONJUNTO = {sp: listar(sp) for sp in ("train", "valid")}
print(f"Entrenamiento: {len(CONJUNTO['train'])} imágenes | validación: {len(CONJUNTO['valid'])}")
problemas = []
for sp, archivos in CONJUNTO.items():
    for a in archivos:
        try:
            leer_caja(sp, a)
        except ValueError as e:
            problemas.append(f"{sp}/{a}: {e}")
if problemas:
    print("\\n".join(problemas[:15]) + ("\\n..." if len(problemas) > 15 else ""))
    raise RuntimeError(f"{len(problemas)} imágenes con problemas en sus etiquetas. Corrígelas y vuelve a armar el conjunto.")
if not CONJUNTO["train"] or not CONJUNTO["valid"]:
    raise RuntimeError("Hace falta al menos una imagen de entrenamiento y una de validación.")
if len(CONJUNTO["train"]) < 100:
    print(f"AVISO: solo {len(CONJUNTO['train'])} imágenes de entrenamiento: probablemente pocas; la variedad de videos pesa más que la cantidad.")

with open(os.path.join(DATASET_DIR, "data.yaml"), encoding="utf-8") as fh:
    print("\\ndata.yaml:\\n" + fh.read())

im0 = cv2.imread(os.path.join(DATASET_DIR, "train", "images", CONJUNTO["train"][0]))
H0, W0 = im0.shape[:2]
anchos = [leer_caja("train", a)[2] * W0 for a in CONJUNTO["train"]]
altos = [leer_caja("train", a)[3] * H0 for a in CONJUNTO["train"]]
print(f"Las imágenes miden {W0}x{H0} px; el ratón mide, en mediana, {np.median(anchos):.0f} x {np.median(altos):.0f} px.")

ruta_ficha = os.path.join(DATASET_DIR, "conjunto_info.json")
FICHA = None
if os.path.exists(ruta_ficha):
    with open(ruta_ficha, encoding="utf-8") as fh:
        FICHA = json.load(fh)
    print("\\nFicha del conjunto:")
    for f in FICHA.get("fuentes", []):
        print(f"  {f.get('prefijo')} ({f.get('rol')}): train {f.get('train')}, valid {f.get('valid')}")
    for a in FICHA.get("advertencias", []):
        print("  AVISO:", a)
else:
    print("\\n(No hay conjunto_info.json: no se sabe de dónde salieron las imágenes.)")

# Mira las imágenes: que el recorte sea la arena, que el ratón se vea grande y que la caja (verde) lo rodee sin la cola
rng = random.Random(0)
muestra = [("train", a) for a in rng.sample(CONJUNTO["train"], min(4, len(CONJUNTO["train"])))] + \\
          [("valid", a) for a in rng.sample(CONJUNTO["valid"], min(4, len(CONJUNTO["valid"])))]
fig, ejes = plt.subplots(2, 4, figsize=(16, 8))
for ax in ejes.ravel():
    ax.axis("off")
for ax, (sp, a) in zip(ejes.ravel(), muestra):
    img = cv2.cvtColor(cv2.imread(os.path.join(DATASET_DIR, sp, "images", a)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    cx, cy, bw, bh = leer_caja(sp, a)
    cv2.rectangle(img, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)), (int((cx + bw / 2) * w), int((cy + bh / 2) * h)), (0, 255, 0), 2)
    ax.imshow(img)
    ax.set_title(f"{sp}/{a}"[:42], fontsize=8)
plt.tight_layout()
plt.show()
'''

PRUEBA = '''
# ============ Paso 6 · Prueba rápida (2 épocas): ¿funciona y cuánto tarda? ============
import gc
from rfdetr import RFDETRNano, RFDETRSmall

CLASE = {"nano": RFDETRNano, "small": RFDETRSmall}[MODELO]


def nuevo_modelo():
    kw = {}
    if RESOLUCION:
        kw["resolution"] = RESOLUCION
    return CLASE(**kw)


def entrenar(modelo, epocas, salida, parada_temprana):
    kw = dict(dataset_dir=DATASET_DIR, epochs=epocas, batch_size=LOTE, grad_accum_steps=ACUMULACION, output_dir=salida)
    if LR:
        kw["lr"] = LR
    extra = {}
    if parada_temprana:
        kw.update(early_stopping=True, early_stopping_patience=PACIENCIA)
        extra = {"skip_best_epochs": EPOCAS_A_IGNORAR}
    try:
        modelo.train(**kw, **extra)
    except (TypeError, ValueError) as e:
        if extra and "skip_best_epochs" in str(e):
            print("Esta versión de rfdetr no acepta skip_best_epochs: se entrena sin ella.")
            modelo.train(**kw)
        else:
            raise
    except torch.cuda.OutOfMemoryError:
        print(f"Se acabó la memoria de la GPU. Baja LOTE (ahora {LOTE}) y sube ACUMULACION para que el producto siga siendo 16; "
              f"si sigue, usa MODELO = 'nano'. Luego reinicia el entorno de ejecución y empieza otra vez.")
        raise


modelo = nuevo_modelo()
RES = getattr(getattr(modelo, "model_config", None), "resolution", None)
print("Resolución del modelo:", RES)
if RES:
    print(f"El ratón mide ~{np.median(anchos) * RES / W0:.0f} x {np.median(altos) * RES / H0:.0f} px a esa resolución.")

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
entrenar(modelo, EPOCAS_PRUEBA, os.path.join(SALIDA, "prueba"), False)
TIEMPO_PRUEBA_S = time.time() - t0
SEG_POR_EPOCA = TIEMPO_PRUEBA_S / EPOCAS_PRUEBA
MEMORIA_PICO_GB = torch.cuda.max_memory_allocated() / 1e9
print(f"\\nPrueba: {TIEMPO_PRUEBA_S:.0f} s en total (~{SEG_POR_EPOCA:.0f} s por época, con el arranque incluido).")
print(f"Cota alta para {EPOCAS} épocas: ~{SEG_POR_EPOCA * EPOCAS / 60:.0f} min (la parada temprana suele acortarlo). Memoria pico de la GPU: {MEMORIA_PICO_GB:.1f} GB.")
print("Los números de mAP de esta prueba no significan nada: solo era para ver que funciona.")
del modelo
gc.collect()
torch.cuda.empty_cache()
'''

ENTRENAR = '''
# ============ Paso 7 · Entrenamiento real ============
modelo = nuevo_modelo()            # vuelve a partir de los pesos preentrenados, no de los de la prueba
SALIDA_ENTRENAMIENTO = os.path.join(SALIDA, "entrenamiento")
t0 = time.time()
entrenar(modelo, EPOCAS, SALIDA_ENTRENAMIENTO, True)
TIEMPO_ENTRENAMIENTO_MIN = (time.time() - t0) / 60
print(f"\\nEntrenamiento terminado en {TIEMPO_ENTRENAMIENTO_MIN:.1f} min. Archivos en {SALIDA_ENTRENAMIENTO}:")
for f in sorted(os.listdir(SALIDA_ENTRENAMIENTO)):
    p = os.path.join(SALIDA_ENTRENAMIENTO, f)
    if os.path.isfile(p):
        print(f"  {f}  ({os.path.getsize(p) / 1e6:.0f} MB)")
del modelo
gc.collect()
torch.cuda.empty_cache()
'''

EVALUAR = '''
# ============ Paso 8 · El mejor punto y su evaluación en validación ============
MEJOR = os.path.join(SALIDA_ENTRENAMIENTO, "checkpoint_best_total.pth")
if not os.path.exists(MEJOR):
    raise FileNotFoundError(f"No encuentro {MEJOR}. Lo que hay: {sorted(os.listdir(SALIDA_ENTRENAMIENTO))}")
mejor = CLASE.from_checkpoint(MEJOR)
metricas = mejor.evaluate(dataset_dir=DATASET_DIR, split="val")
METRICAS = {k: float(v) for k, v in metricas.items()}
print("Métricas en la validación (mAP = qué tanto coinciden las cajas con las del conjunto):")
for k, v in sorted(METRICAS.items()):
    print(f"  {k}: {v:.4f}")
'''

COMPARAR = '''
# ============ Paso 9 · Cajas del detector contra las del conjunto (validación) ============
def iou_xyxy(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def cargar(split, archivo):
    """Imagen RGB y caja (x1, y1, x2, y2) en píxeles, del conjunto."""
    img = cv2.cvtColor(cv2.imread(os.path.join(DATASET_DIR, split, "images", archivo)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    cx, cy, bw, bh = leer_caja(split, archivo)
    return img, ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)


def espaciadas(lista, n):
    paso = max(1, len(lista) // n)
    return lista[::paso][:n]


filas, ejemplos = [], []
for a in espaciadas(CONJUNTO["valid"], 200):
    img, gt = cargar("valid", a)
    det = mejor.predict(img, threshold=UMBRAL)
    if len(det.xyxy) == 0:
        filas.append(None)
        continue
    k = int(np.argmax(det.confidence))
    pred = [float(v) for v in det.xyxy[k]]
    centro = ((pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2)
    filas.append((iou_xyxy(pred, gt), (pred[2] - pred[0]) / (gt[2] - gt[0]), (pred[3] - pred[1]) / (gt[3] - gt[1]),
                  gt[0] <= centro[0] <= gt[2] and gt[1] <= centro[1] <= gt[3]))
    if len(ejemplos) < 8:
        ejemplos.append((img, gt, pred, float(det.confidence[k]), a))

hallazgos = [f for f in filas if f is not None]
RESUMEN_TORCH = {"imagenes": len(filas), "con_deteccion": len(hallazgos)}
print(f"Se detectó un ratón en {len(hallazgos)} de {len(filas)} imágenes de validación (umbral {UMBRAL}).")
if hallazgos:
    ious = [f[0] for f in hallazgos]
    RESUMEN_TORCH.update(iou_mediana=float(np.median(ious)), iou_p10=float(np.percentile(ious, 10)),
                         ancho_razon_mediana=float(np.median([f[1] for f in hallazgos])), alto_razon_mediana=float(np.median([f[2] for f in hallazgos])),
                         centro_dentro_pct=100.0 * sum(f[3] for f in hallazgos) / len(hallazgos))
    print(f"IoU mediano {RESUMEN_TORCH['iou_mediana']:.2f} (el 10 % peor: {RESUMEN_TORCH['iou_p10']:.2f}) | centro dentro de la caja: {RESUMEN_TORCH['centro_dentro_pct']:.0f} %")
    print(f"La caja del detector es, en mediana, {RESUMEN_TORCH['ancho_razon_mediana']:.2f} veces el ancho y {RESUMEN_TORCH['alto_razon_mediana']:.2f} veces el alto de la del conjunto.")
print("Recuerda: son videos de validación con etiquetas del conjunto, no la evaluación final.")

fig, ejes = plt.subplots(2, 4, figsize=(16, 8))
for ax in ejes.ravel():
    ax.axis("off")
for ax, (img, gt, pred, conf, a) in zip(ejes.ravel(), ejemplos):
    im = img.copy()
    cv2.rectangle(im, (int(gt[0]), int(gt[1])), (int(gt[2]), int(gt[3])), (255, 255, 255), 2)   # blanca: la del conjunto
    cv2.rectangle(im, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 255, 0), 2)  # verde: la del detector
    ax.imshow(im)
    ax.set_title(f"{a[:30]}  conf {conf:.2f}", fontsize=8)
plt.tight_layout()
plt.show()
'''

EXPORTAR = '''
# ============ Paso 10 · Exportar a ONNX ============
NOTAS = {
    "clase": "raton", "modelo": MODELO, "resolucion": RES, "version_rfdetr": VERSION_INSTALADA, "umbral_sugerido": UMBRAL,
    "recorte_roi": FICHA.get("recorte_roi") if FICHA else None, "margen_roi_px": FICHA.get("margen_roi_px") if FICHA else None,
    "imagenes": FICHA.get("imagenes") if FICHA else None, "corrida": NOMBRE_CORRIDA,
}
carpeta_onnx = os.path.join(SALIDA, "onnx")
try:
    RUTA_ONNX = str(mejor.export(output_dir=carpeta_onnx, notes=NOTAS))    # las notas quedan dentro del archivo ONNX
except TypeError as e:
    if "notes" not in str(e):
        raise
    print("Esta versión de rfdetr no acepta notes: se exporta sin metadatos.")
    RUTA_ONNX = str(mejor.export(output_dir=carpeta_onnx))
if not os.path.exists(RUTA_ONNX):
    raise FileNotFoundError(f"La exportación no dejó el archivo {RUTA_ONNX}")
print(f"ONNX: {RUTA_ONNX}  ({os.path.getsize(RUTA_ONNX) / 1e6:.0f} MB)")
'''

VERIFICAR = '''
# ============ Paso 11 · ¿El ONNX hace lo mismo que el modelo original? ============
import onnxruntime as ort

sesion = ort.InferenceSession(RUTA_ONNX, providers=["CPUExecutionProvider"])
print("Entrada del grafo:", sesion.get_inputs()[0].name, sesion.get_inputs()[0].shape, "| salidas:", [o.name for o in sesion.get_outputs()])

coinciden, distintos, ious_onnx, tiempos = 0, 0, [], []
for a in espaciadas(CONJUNTO["valid"], 50):
    img, gt = cargar("valid", a)
    t = time.perf_counter()
    d_onnx = detect(sesion, img, UMBRAL)
    tiempos.append(time.perf_counter() - t)
    d_torch = mejor.predict(img, threshold=UMBRAL)
    if len(d_onnx.scores) == 0 and len(d_torch.xyxy) == 0:
        coinciden += 1
    elif len(d_onnx.scores) == 0 or len(d_torch.xyxy) == 0:
        distintos += 1
    else:
        k = int(np.argmax(d_torch.confidence))
        ious_onnx.append(iou_xyxy([float(v) for v in d_onnx.xyxy[0]], [float(v) for v in d_torch.xyxy[k]]))
VERIF_ONNX = {"imagenes": coinciden + distintos + len(ious_onnx), "uno_detecta_y_otro_no": distintos,
              "ms_por_imagen_en_la_cpu_de_colab": 1000 * float(np.median(tiempos))}
if ious_onnx:
    VERIF_ONNX.update(iou_mediana_onnx_vs_torch=float(np.median(ious_onnx)), iou_minimo=float(np.min(ious_onnx)))
    print(f"ONNX contra el modelo original: IoU mediano {VERIF_ONNX['iou_mediana_onnx_vs_torch']:.3f}, el peor {VERIF_ONNX['iou_minimo']:.3f}.")
print(f"Discrepancias (uno detecta y el otro no): {distintos} de {VERIF_ONNX['imagenes']}. Tiempo en la CPU de Colab: ~{VERIF_ONNX['ms_por_imagen_en_la_cpu_de_colab']:.0f} ms por imagen "
      f"(solo de referencia: en tu Mac será otro).")
if distintos > 0.1 * VERIF_ONNX["imagenes"] or (ious_onnx and np.median(ious_onnx) < 0.9):
    print("ADVERTENCIA: el ONNX no coincide bien con el modelo original. No lo uses sin revisarlo (¿ranura de fondo, resolución, normalización?).")
else:
    print("El ONNX coincide con el modelo original.")
'''

GUARDAR = '''
# ============ Paso 12 · Guardar y descargar el resultado ============
os.makedirs(ENTREGA, exist_ok=True)
entregables = []


def copiar(origen, nombre=None):
    if origen and os.path.exists(origen):
        destino = os.path.join(ENTREGA, nombre or os.path.basename(origen))
        shutil.copyfile(origen, destino)
        entregables.append(destino)


copiar(MEJOR, "checkpoint_best_total.pth")
copiar(RUTA_ONNX)
copiar(os.path.join(SALIDA_ENTRENAMIENTO, "training_config.json"))
copiar(os.path.join(DATASET_DIR, "conjunto_info.json"))

RUN_INFO = {
    "corrida": NOMBRE_CORRIDA, "fecha": datetime.now().isoformat(timespec="seconds"), "modelo": MODELO, "version_rfdetr": VERSION_INSTALADA,
    "resolucion": RES, "gpu": GPU, "vram_gb": round(VRAM_GB, 1),
    "parametros": {"epocas": EPOCAS, "lote": LOTE, "acumulacion": ACUMULACION, "lr": LR, "paciencia": PACIENCIA, "epocas_a_ignorar": EPOCAS_A_IGNORAR, "umbral": UMBRAL},
    "tiempos": {"prueba_s": round(TIEMPO_PRUEBA_S, 1), "entrenamiento_min": round(TIEMPO_ENTRENAMIENTO_MIN, 1)},
    "memoria_pico_gb_en_la_prueba": round(MEMORIA_PICO_GB, 1),
    "metricas_validacion": METRICAS, "comparacion_en_validacion": RESUMEN_TORCH, "verificacion_onnx": VERIF_ONNX,
    "notas_en_el_onnx": NOTAS, "conjunto": {"imagenes": FICHA.get("imagenes") if FICHA else None, "advertencias": FICHA.get("advertencias") if FICHA else None},
    "aviso": "La validación de este cuaderno no es la evaluación final: esa se hace en el Mac, con el video reservado y etiquetas a mano.",
}
ruta_info = os.path.join(ENTREGA, "run_info.json")
with open(ruta_info, "w", encoding="utf-8") as fh:
    json.dump(RUN_INFO, fh, indent=2, ensure_ascii=False)
entregables.append(ruta_info)

ZIP = os.path.join(ENTREGA, NOMBRE_CORRIDA + ".zip")
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for e in entregables:
        z.write(e, os.path.basename(e))
print("Guardado en", ENTREGA)
for e in entregables + [ZIP]:
    print(f"  {os.path.basename(e)}  ({os.path.getsize(e) / 1e6:.1f} MB)")

if not GUARDAR_EN_DRIVE:
    from google.colab import files
    files.download(ZIP)          # sin Drive, descárgalo ahora: lo local se pierde al desconectarse
'''

SIGUE = """
## Qué sigue

1. Lleva `checkpoint_best_total.pth`, el `.onnx` y `run_info.json` a tu Mac (`TrackerLab_Resultados/modelos/<corrida>/`). **Anota la versión de rfdetr** que quedó en `run_info.json`.
2. Evalúa el modelo en tu Mac con el **video reservado** y tus 20 etiquetas a mano, con las mismas métricas de siempre (IoU ≥ 0.5, posición dentro de la caja, error de centro). Esa es la medida que cuenta.
3. Si entrenaste `small`, entrena también `nano` con los mismos datos y compara exactitud y velocidad **en tu Mac**.

## Si algo falla

| Qué ves | Qué hacer |
|---|---|
| «No hay GPU» | Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU, y empieza de nuevo |
| Error de memoria de la GPU | Baja `LOTE` y sube `ACUMULACION` (el producto sigue siendo 16), o usa `MODELO = "nano"`; reinicia el entorno |
| Colab se desconectó a mitad del entrenamiento | Lo local se perdió; vuelve a empezar con menos `EPOCAS`. Lo copiado a Drive sigue ahí |
| El entrenamiento no mejora | Primero revisa las imágenes del paso 5. Después prueba `LR` a la mitad. No ajustes nada más hasta ver una corrida |
| «ADVERTENCIA: el ONNX no coincide» | No uses ese ONNX. Avísame con el mensaje completo: puede ser la ranura de fondo o la resolución |
| `skip_best_epochs` no se reconoce | No pasa nada: el cuaderno entrena sin esa opción (usa una versión más antigua de rfdetr) |
"""


def build_notebook() -> Dict[str, Any]:
    cells = [
        md(INTRO),
        code(CONFIG),
        md("## Paso 1 · Instalar RF-DETR\nTarda un par de minutos. Si Colab pide reiniciar el entorno de ejecución, hazlo y vuelve a empezar desde la celda de configuración."),
        code(INSTALL),
        md("## Paso 2 · Comprobar la GPU"),
        code(GPU),
        md("## Paso 3 · Carpetas de trabajo y Drive\nPide permiso para conectar tu Drive (solo se usa para guardar el resultado final)."),
        code(DRIVE),
        md("## Paso 4 · Traer el conjunto de datos\nSube `entrenamiento.zip` con el botón que aparece, o indica su ruta en Drive con `ZIP_EN_DRIVE`."),
        code(DATOS),
        md("## Paso 5 · Revisar el conjunto antes de entrenar\nComprueba que todas las etiquetas están bien y muestra 8 imágenes con su caja. **Míralas**: el recorte debe ser la arena, el ratón debe verse grande y la caja (verde) debe rodear el cuerpo sin la cola. Si algo se ve mal, no sigas: avísame."),
        code(REVISAR),
        md("## Decodificación del ONNX\nEste código es el mismo que usará TrackerLab (`learning/onnx_decode.py`); el cuaderno lo trae incrustado."),
        code("# ---- learning/onnx_decode.py (fuente única; no editar aquí) ----\n" + decode_source()),
        md("## Paso 6 · Prueba rápida de 2 épocas\nMide cuánto tarda y cuánta memoria usa. Sus resultados no significan nada: solo confirma que todo funciona."),
        code(PRUEBA),
        md("## Paso 7 · Entrenamiento real\nSe detiene solo si deja de mejorar. Puedes interrumpirlo (■) si necesitas parar, pero lo entrenado hasta ahí se pierde si no llegó a guardar un mejor punto."),
        code(ENTRENAR),
        md("## Paso 8 · El mejor punto y su evaluación"),
        code(EVALUAR),
        md("## Paso 9 · Cajas del detector contra las del conjunto\nBlanca = la del conjunto; verde = la del detector. Fíjate sobre todo en si la verde es más angosta o incluye la cola."),
        code(COMPARAR),
        md("## Paso 10 · Exportar a ONNX"),
        code(EXPORTAR),
        md("## Paso 11 · Verificar el ONNX\nEl ONNX devuelve datos sin decodificar; esta celda comprueba que, decodificados, dan lo mismo que el modelo original."),
        code(VERIFICAR),
        md("## Paso 12 · Guardar y descargar"),
        code(GUARDAR),
        md(SIGUE),
    ]
    return {
        "cells": cells,
        "metadata": {
            "colab": {"name": "entrenar_rfdetr.ipynb", "provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def write_notebook(path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(build_notebook(), fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def main(argv: Optional[List[str]] = None, out=print) -> int:
    ap = argparse.ArgumentParser(description="Genera el cuaderno de Colab que entrena el detector de ratones (RF-DETR).")
    ap.add_argument("--salida", default=DEFAULT_OUT, help="ruta del .ipynb (por omisión colab/entrenar_rfdetr.ipynb)")
    a = ap.parse_args(argv)
    try:
        ruta = write_notebook(a.salida)
    except (OSError, RuntimeError) as exc:
        out(str(exc))
        return 2
    nb = build_notebook()
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    out(f"Cuaderno escrito en {ruta} ({len(nb['cells'])} celdas, {n_code} de código).")
    out("Súbelo a Colab (colab.research.google.com → Archivo → Subir cuaderno) y ejecútalo de arriba abajo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())