from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
from collections import Counter, deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

__all__ = [
    "CLASS_NAMES",
    "yolo_label_line",
    "parse_yolo_label_line",
    "split_for_time",
    "video_fingerprint",
    "DatasetCollector",
    "make_audit_sheet",
]

CLASS_NAMES = ("mouse",)
LABEL_ORIGIN = "seguimiento_clasico"
Box = Tuple[float, float, float, float]  # (x, y, w, h) en píxeles


# Formato YOLO
def yolo_label_line(bbox: Box, frame_w: int, frame_h: int, cls: int = 0) -> str:
    """Línea de etiqueta YOLO: ``clase cx cy ancho alto``, normalizada a 0..1 y recortada al frame."""
    x, y, w, h = (float(v) for v in bbox)
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(frame_w), x + w), min(float(frame_h), y + h)
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        raise ValueError(f"La caja {bbox} queda fuera del frame {frame_w}x{frame_h}.")
    cx, cy = (x0 + x1) / 2.0 / frame_w, (y0 + y1) / 2.0 / frame_h
    return f"{int(cls)} {cx:.6f} {cy:.6f} {(x1 - x0) / frame_w:.6f} {(y1 - y0) / frame_h:.6f}"


def parse_yolo_label_line(line: str, frame_w: int, frame_h: int) -> Tuple[int, Box]:
    """Inversa de ``yolo_label_line``: ``(clase, (x, y, w, h))`` en píxeles."""
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"Etiqueta YOLO inválida: {line!r}")
    cls, cx, cy, w, h = int(parts[0]), *(float(v) for v in parts[1:])
    return cls, ((cx - w / 2.0) * frame_w, (cy - h / 2.0) * frame_h, w * frame_w, h * frame_h)


# Reparto entrenamiento / validación

def split_for_time(time_s: float, block_seconds: float = 20.0, val_every: int = 5, buffer_s: float = 1.0) -> Optional[str]:
    if val_every < 2 or block_seconds <= 0 or buffer_s < 0 or 2 * buffer_s >= block_seconds:
        raise ValueError("Parámetros de reparto inválidos.")
    block = int(time_s // block_seconds)
    t_in = time_s - block * block_seconds
    is_val = block % val_every == val_every - 1
    prev_val = block > 0 and (block - 1) % val_every == val_every - 1
    next_val = (block + 1) % val_every == val_every - 1
    if is_val:
        return None if (t_in < buffer_s or t_in > block_seconds - buffer_s) else "val"
    if (prev_val and t_in < buffer_s) or (next_val and t_in > block_seconds - buffer_s):
        return None
    return "train"


def video_fingerprint(path: str) -> Dict[str, Any]:
    """Huella rápida de un video: tamaño y SHA-256 de su primer y último megabyte (no lee el archivo entero)."""
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        if size > 2 << 20:
            f.seek(-(1 << 20), os.SEEK_END)
            h.update(f.read(1 << 20))
    return {"size_bytes": size, "sha256_extremos": h.hexdigest()}


# Recolector
class DatasetCollector:

    def __init__(
        self,
        out_dir: str,
        video_name: str = "video",
        *,
        stride: int = 5,
        min_stable_frames: int = 10,
        area_ratio: Tuple[float, float] = (0.5, 2.0),
        border_px: int = 2,
        block_seconds: float = 20.0,
        val_every: int = 5,
        buffer_s: float = 1.0,
        jpeg_quality: int = 92,
        max_images: Optional[int] = None,
        tracker_mode: str = "",
        video_path: Optional[str] = None,
    ):
        if stride < 1 or min_stable_frames < 1:
            raise ValueError("stride y min_stable_frames deben ser >= 1")
        split_for_time(0.0, block_seconds, val_every, buffer_s)  # valida los parámetros del reparto
        self.out_dir, self.video_name = out_dir, video_name
        self.stride, self.min_stable = stride, min_stable_frames
        self.area_ratio, self.border_px = area_ratio, border_px
        self.block_seconds, self.val_every, self.buffer_s = block_seconds, val_every, buffer_s
        self.jpeg_quality, self.max_images = jpeg_quality, max_images
        self.tracker_mode, self.video_path = tracker_mode, video_path
        for split in ("train", "val"):
            os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)
        self._stable = 0
        self._eligible = 0
        self._areas: Deque[float] = deque(maxlen=50)
        self._skipped: Counter = Counter()
        self._rows: List[Dict[str, Any]] = []
        self._frame_size: Optional[Tuple[int, int]] = None

    def _skip(self, reason: str) -> None:
        self._skipped[reason] += 1

    def __call__(self, event) -> None:
        """Se usa como ``on_frame``. Siempre devuelve None (no interfiere con las órdenes de la interfaz)."""
        rec, det = event.record, event.detection
        if not (rec.status == "detectado" and rec.found and det.found):
            self._stable = 0
            return self._skip("no_detectado")
        self._stable += 1
        if self._stable < self.min_stable:
            return self._skip("inestable")

        x0, y0, w0, h0 = event.roi_box
        bx, by, bw, bh = det.bbox
        m = self.border_px
        if bx < m or by < m or bx + bw > w0 - m or by + bh > h0 - m:
            return self._skip("borde")
        area = float(bw * bh)
        if len(self._areas) >= 10:
            med = statistics.median(self._areas)
            if not (self.area_ratio[0] * med <= area <= self.area_ratio[1] * med):
                return self._skip("area")  # no entra al historial: así una caja rara no desplaza la referencia
        self._areas.append(area)

        self._eligible += 1
        if (self._eligible - 1) % self.stride != 0:
            return self._skip("muestreo")
        split = split_for_time(rec.time_s, self.block_seconds, self.val_every, self.buffer_s)
        if split is None:
            return self._skip("separacion")
        if self.max_images is not None and len(self._rows) >= self.max_images:
            return self._skip("limite")

        frame = event.frame
        fh, fw = frame.shape[:2]
        if self._frame_size is None:
            self._frame_size = (fw, fh)
        elif self._frame_size != (fw, fh):
            raise ValueError("El tamaño del frame cambió durante la corrida.")
        box = (bx + x0, by + y0, bw, bh)  # del recorte del ROI al frame completo
        line = yolo_label_line(box, fw, fh)

        name = f"{self.video_name}_{event.frame_idx:07d}"
        img_path = os.path.join(self.out_dir, "images", split, name + ".jpg")
        if not cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, int(self.jpeg_quality)]):
            raise OSError(f"No se pudo escribir {img_path}")
        with open(os.path.join(self.out_dir, "labels", split, name + ".txt"), "w", encoding="utf-8") as f:
            f.write(line + "\n")
        self._rows.append({"archivo": f"{split}/{name}", "split": split, "frame": event.frame_idx,
                           "time_s": round(rec.time_s, 6), "x": box[0], "y": box[1], "w": box[2], "h": box[3]})

    def finalize(self) -> Dict[str, Any]:
        """Escribe ``data.yaml``, ``manifest.csv`` y ``dataset_info.json``. Devuelve el contenido de este último."""
        with open(os.path.join(self.out_dir, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["archivo", "split", "frame", "time_s", "x", "y", "w", "h"])
            w.writeheader()
            w.writerows(self._rows)
        names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
        with open(os.path.join(self.out_dir, "data.yaml"), "w", encoding="utf-8") as f:
            f.write(f"path: {json.dumps(os.path.abspath(self.out_dir))}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
        counts = Counter(r["split"] for r in self._rows)
        info: Dict[str, Any] = {
            "creado": datetime.now().isoformat(timespec="seconds"),
            "origen_de_las_etiquetas": LABEL_ORIGIN,
            "aviso": ("Las etiquetas son AUTOMÁTICAS (las produjo el seguimiento clásico) y heredan sus errores. "
                      "Revisa una muestra con make_audit_sheet antes de entrenar. Evaluar un detector contra estas "
                      "mismas etiquetas es circular; para medir de verdad hace falta validación etiquetada a mano."),
            "modo_de_seguimiento": self.tracker_mode,
            "video": {"nombre": self.video_name, **(video_fingerprint(self.video_path) if self.video_path and os.path.exists(self.video_path) else {})},
            "parametros": {"stride": self.stride, "min_stable_frames": self.min_stable, "area_ratio": list(self.area_ratio),
                           "border_px": self.border_px, "block_seconds": self.block_seconds, "val_every": self.val_every,
                           "buffer_s": self.buffer_s, "jpeg_quality": self.jpeg_quality, "max_images": self.max_images},
            "imagenes": {"train": counts.get("train", 0), "val": counts.get("val", 0), "total": len(self._rows)},
            "frames_descartados_por_motivo": dict(self._skipped),
            "clases": list(CLASS_NAMES),
        }
        with open(os.path.join(self.out_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        return info


# Revisión visual
def make_audit_sheet(dataset_dir: str, out_path: str, n: int = 24, cols: int = 6, thumb_w: int = 320, seed: int = 0) -> Optional[str]:
    """Hoja de contactos con ``n`` ejemplos al azar y su caja dibujada, para revisar las etiquetas a simple vista.

    Devuelve la ruta guardada, o None si el conjunto está vacío.
    """
    items = []
    for split in ("train", "val"):
        ldir = os.path.join(dataset_dir, "labels", split)
        if os.path.isdir(ldir):
            items += [(split, f[:-4]) for f in sorted(os.listdir(ldir)) if f.endswith(".txt")]
    if not items:
        return None
    rng = np.random.default_rng(seed)
    chosen = [items[i] for i in sorted(rng.choice(len(items), size=min(n, len(items)), replace=False))]
    tiles = []
    for split, name in chosen:
        img = cv2.imread(os.path.join(dataset_dir, "images", split, name + ".jpg"))
        if img is None:
            continue
        h, w = img.shape[:2]
        with open(os.path.join(dataset_dir, "labels", split, name + ".txt"), encoding="utf-8") as f:
            _, (x, y, bw, bh) = parse_yolo_label_line(f.readline(), w, h)
        cv2.rectangle(img, (int(round(x)), int(round(y))), (int(round(x + bw)), int(round(y + bh))), (0, 255, 0), max(2, w // 300))
        scale = thumb_w / w
        tile = cv2.resize(img, (thumb_w, int(round(h * scale))))
        cv2.putText(tile, f"{split} {name[-7:]}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return None
    th = max(t.shape[0] for t in tiles)
    cols = max(1, min(cols, len(tiles)))
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    canvas_rows = []
    for r in rows:
        r = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30)) for t in r]
        r += [np.full((th, thumb_w, 3), 30, np.uint8)] * (cols - len(r))
        canvas_rows.append(np.hstack(r))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, np.vstack(canvas_rows))
    return out_path