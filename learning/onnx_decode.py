"""Decodificación de la salida ONNX de RF-DETR: código puro (solo numpy y cv2), sin onnxruntime ni torch.

Es la fuente ÚNICA de esta lógica: la usa el cuaderno de Colab (que la trae incrustada) y la usará el detector de TrackerLab.

Según la documentación oficial de RF-DETR (sección «ONNX Inference»), el grafo exportado NO decodifica nada:
  * entrada «input»: imagen RGB, escalada a 0-1, normalizada con la media y la desviación de ImageNet, formato NCHW, y
    redimensionada al tamaño de entrada del grafo;
  * salida «dets»: cajas ``pred_boxes`` normalizadas en formato (cx, cy, ancho, alto);
  * salida «labels»: ``pred_logits`` SIN activar, con una ranura más que las clases (la de fondo).
Hay que: aplicar la sigmoide por clase (no softmax), quitar la ranura de fondo si la hay, y convertir las cajas a píxeles.

Dos trampas que la documentación señala y este código cubre:
  * Las salidas se identifican POR NOMBRE, nunca por forma ni por posición: con 3 clases, las dos salidas tienen ``4``
    como última dimensión y se confundirían.
  * La ranura de fondo depende de cómo se entrenó: en un conjunto propio con clases 0, 1, 2... es la ÚLTIMA (índice -1),
    que es nuestro caso (una sola clase «raton»: 2 ranuras, la 0 es el ratón y la 1 es el fondo). En los pesos oficiales
    de COCO no hay ranura de fondo (``background_class_id=None``).
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional, Sequence, Tuple

import numpy as np

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class Detections(NamedTuple):
    """Detecciones ordenadas de mayor a menor confianza. Las cajas son (x1, y1, x2, y2) en píxeles de la imagen original."""

    xyxy: np.ndarray  # (n, 4) float32
    scores: np.ndarray  # (n,) float32
    class_ids: np.ndarray  # (n,) int64


def sigmoid(x: Any) -> np.ndarray:
    """Sigmoide sin desbordes (el argumento se limita a ±88)."""
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=np.float64), -88.0, 88.0)))


def preprocess(image_rgb: np.ndarray, input_hw: Tuple[int, int]) -> np.ndarray:
    """Imagen RGB (uint8, alto x ancho x 3) -> tensor float32 (1, 3, alto, ancho) listo para el grafo."""
    import cv2

    img = np.asarray(image_rgb)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Se esperaba una imagen RGB de alto x ancho x 3, no {img.shape}.")
    if img.dtype != np.uint8:
        raise ValueError(f"Se esperaba una imagen uint8 (0-255), no {img.dtype}.")
    h, w = int(input_hw[0]), int(input_hw[1])
    resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    x = (resized.astype(np.float32) / 255.0 - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def find_outputs(names: Sequence[str]) -> Tuple[int, int]:
    """Posición de las cajas («dets») y de los logits («labels») entre las salidas del grafo, buscándolas por nombre."""
    boxes = next((i for i, n in enumerate(names) if "dets" in n), None)
    logits = next((i for i, n in enumerate(names) if "labels" in n), None)
    if boxes is None or logits is None:
        raise ValueError(f"No se encontraron las salidas «dets» y «labels» entre {list(names)}.")
    if boxes == logits:
        raise ValueError(f"La salida {names[boxes]!r} no puede ser a la vez las cajas y los logits.")
    return boxes, logits


def decode(
    boxes_cxcywh: Any,
    raw_logits: Any,
    image_w: int,
    image_h: int,
    threshold: float = 0.5,
    background_class_id: Optional[int] = -1,
    clip: bool = True,
) -> Detections:
    """Cajas normalizadas (cx, cy, w, h) y logits sin activar -> detecciones en píxeles de la imagen original.

    background_class_id: índice de la ranura de fondo (-1 = la última, nuestro caso); None si el modelo no tiene (COCO oficial).
    Una detección se conserva si su confianza es ESTRICTAMENTE mayor que ``threshold``.
    """
    boxes = np.asarray(boxes_cxcywh, dtype=np.float64)
    logits = np.asarray(raw_logits, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Las cajas deben ser (n, 4), no {boxes.shape}.")
    if logits.ndim != 2 or logits.shape[0] != boxes.shape[0]:
        raise ValueError(f"Los logits {logits.shape} no corresponden a las cajas {boxes.shape}.")
    slots = np.arange(logits.shape[1])
    if background_class_id is not None:
        n = logits.shape[1]
        if not -n <= background_class_id < n:
            raise ValueError(f"background_class_id={background_class_id} no es una ranura válida de {n}.")
        keep = slots != (background_class_id % n)
        logits, slots = logits[:, keep], slots[keep]
    if logits.shape[1] == 0:
        return Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))
    act = sigmoid(logits)  # sigmoide por clase (multi-etiqueta), no softmax
    scores, ids = act.max(axis=1), slots[act.argmax(axis=1)]
    sel = scores > threshold
    cx, cy, bw, bh = boxes[sel].T
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1) * np.array([image_w, image_h, image_w, image_h], dtype=np.float64)
    if clip:
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, image_w)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, image_h)
    orden = np.argsort(-scores[sel], kind="stable")
    return Detections(xyxy[orden].astype(np.float32), scores[sel][orden].astype(np.float32), ids[sel][orden].astype(np.int64))


def detect(session: Any, image_rgb: np.ndarray, threshold: float = 0.5, background_class_id: Optional[int] = -1) -> Detections:
    """Corre una sesión de onnxruntime (o algo con la misma interfaz) sobre UNA imagen RGB y devuelve sus detecciones."""
    entrada = session.get_inputs()[0]
    forma = list(entrada.shape)
    if len(forma) != 4 or not all(isinstance(v, int) and v > 0 for v in forma[2:4]):
        raise ValueError(f"El grafo debe tener una entrada de tamaño fijo (1, 3, alto, ancho); la suya es {forma}.")
    x = preprocess(image_rgb, (forma[2], forma[3]))
    salidas = session.run(None, {entrada.name: x})
    ib, il = find_outputs([o.name for o in session.get_outputs()])
    h, w = np.asarray(image_rgb).shape[:2]
    return decode(np.asarray(salidas[ib])[0], np.asarray(salidas[il])[0], w, h, threshold, background_class_id)