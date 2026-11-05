from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learning.dataset import video_fingerprint  # noqa: E402

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv")
N_FRAMES = 5  # frames repartidos para estimar el fondo
SIG_SIZE = (64, 36)  # (ancho, alto) de la firma del fondo
DRIVE_MARCAS = ("CloudStorage", "GoogleDrive", "Google Drive", "My Drive", "Mi unidad")


# Piezas puras
def fmt_duration(s: Optional[float]) -> str:
    if s is None or s != s or s < 0:
        return "?"
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def looks_like_drive(path: str) -> bool:
    return any(m in path for m in DRIVE_MARCAS)


def scan(carpetas: Sequence[str]) -> List[str]:
    """Videos dentro de las carpetas (recursivo), sin archivos ocultos y sin repetir; ordenados por ruta."""
    vistos, out = set(), []
    for c in carpetas:
        if os.path.isfile(c):
            candidatos = [c]
        else:
            candidatos = []
            for raiz, dirs, files in os.walk(c):
                dirs[:] = sorted(d for d in dirs if not d.startswith("."))
                candidatos += [os.path.join(raiz, f) for f in sorted(files)]
        for p in candidatos:
            if os.path.basename(p).startswith(".") or not p.lower().endswith(VIDEO_EXTS):
                continue
            real = os.path.realpath(p)
            if real not in vistos:
                vistos.add(real)
                out.append(p)
    return sorted(out)


def sample_positions(total_frames: int, n: int = N_FRAMES, lo: float = 0.1, hi: float = 0.9) -> List[int]:
    """``n`` posiciones de frame repartidas entre el ``lo`` y el ``hi`` del video (sin los extremos, que suelen ser oscuros)."""
    if total_frames < 1:
        return []
    if total_frames <= n:
        return list(range(total_frames))
    return sorted({min(total_frames - 1, int(total_frames * (lo + (hi - lo) * i / (n - 1)))) for i in range(n)})


def background_signature(frames: Sequence[np.ndarray], size: Tuple[int, int] = SIG_SIZE) -> Optional[np.ndarray]:
    """Mediana, píxel a píxel, de los frames reducidos y en gris: el animal se mueve y casi desaparece; el fondo se queda."""
    if not frames:
        return None
    pequenos = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f, size, interpolation=cv2.INTER_AREA) for f in frames]
    return np.median(np.stack(pequenos).astype(np.float32), axis=0)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Correlación de Pearson entre dos firmas (1 = mismo patrón; no depende del brillo medio ni del contraste global)."""
    x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    sx, sy = x.std(), y.std()
    if sx < 1e-9 or sy < 1e-9:
        return 1.0 if (sx < 1e-9 and sy < 1e-9 and abs(x.mean() - y.mean()) < 1e-6) else 0.0
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))


def group_by_similarity(sigs: Sequence[Optional[np.ndarray]], umbral: float) -> Tuple[List[Optional[int]], np.ndarray]:
    """Grupos por componentes conexas (dos videos se unen si su correlación >= umbral). Devuelve (grupo por video, matriz).

    Los videos sin firma (None) no pertenecen a ningún grupo. Los grupos se numeran por orden de aparición, desde 1.
    """
    n = len(sigs)
    m = np.full((n, n), np.nan)
    padre = list(range(n))

    def raiz(i: int) -> int:
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    for i in range(n):
        if sigs[i] is None:
            continue
        m[i, i] = 1.0
        for j in range(i + 1, n):
            if sigs[j] is None:
                continue
            c = correlation(sigs[i], sigs[j])  # type: ignore[arg-type]
            m[i, j] = m[j, i] = c
            if c >= umbral - 1e-9:  # tolerancia: dos firmas idénticas pueden dar 0.9999999999999999
                padre[raiz(j)] = raiz(i)
    numero: Dict[int, int] = {}
    grupos: List[Optional[int]] = []
    for i in range(n):
        if sigs[i] is None:
            grupos.append(None)
            continue
        r = raiz(i)
        numero.setdefault(r, len(numero) + 1)
        grupos.append(numero[r])
    return grupos, m


# Lectura
@dataclass
class Video:
    id: str
    ruta: str
    tam_bytes: int
    modificado: str
    abierto: bool = False
    error: str = ""
    fps: Optional[float] = None
    frames: Optional[int] = None
    ancho: Optional[int] = None
    alto: Optional[int] = None
    duracion_s: Optional[float] = None
    huella: str = ""
    duplicado_de: str = ""
    grupo: Optional[int] = None
    firma: Optional[np.ndarray] = field(default=None, repr=False)
    miniatura: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def nombre(self) -> str:
        return os.path.basename(self.ruta)


def read_video(v: Video) -> None:
    """Rellena metadatos, huella, firma del fondo y miniatura. Nunca lanza: los problemas quedan en ``v.error``.

    La huella se calcula siempre (aunque el video no abra), porque sirve para detectar duplicados.
    """
    cap = cv2.VideoCapture(v.ruta)
    try:
        if not cap.isOpened():
            v.error = "no se pudo abrir"
        else:
            v.abierto = True
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            v.ancho, v.alto = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
            v.fps = fps if fps > 0 else None
            v.frames = total if total > 0 else None
            v.duracion_s = total / fps if fps > 0 and total > 0 else None
            if total <= 0:
                v.error = "el archivo no informa cuántos frames tiene: no se pudo muestrear"
            else:
                tomados: List[np.ndarray] = []
                for pos in sample_positions(total):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                    ok, f = cap.read()
                    if ok and f is not None:
                        tomados.append(f)
                if not tomados:
                    v.error = "no se pudo leer ningún frame"
                else:
                    v.firma = background_signature(tomados)
                    v.miniatura = tomados[len(tomados) // 2]
    except Exception as exc:  # un video raro no debe tirar el inventario
        v.error = f"error al leer: {exc}"
    finally:
        cap.release()
    try:
        v.huella = video_fingerprint(v.ruta)["sha256_extremos"]
    except OSError as exc:
        v.error = (v.error + "; " if v.error else "") + f"sin huella: {exc}"


def mark_duplicates(videos: Sequence[Video]) -> None:
    vistos: Dict[Tuple[str, int], str] = {}
    for v in videos:
        if not v.huella:
            continue
        clave = (v.huella, v.tam_bytes)
        if clave in vistos:
            v.duplicado_de = vistos[clave]
        else:
            vistos[clave] = v.id


# Salidas
def contact_sheet(videos: Sequence[Video], cols: int = 4, thumb_w: int = 320) -> Optional[np.ndarray]:
    if not videos:
        return None
    thumb_h, banda = thumb_w * 9 // 16, 34
    filas = (len(videos) + cols - 1) // cols
    hoja = np.full((filas * (thumb_h + banda), cols * thumb_w, 3), 30, np.uint8)
    for k, v in enumerate(videos):
        r, c = divmod(k, cols)
        y0, x0 = r * (thumb_h + banda), c * thumb_w
        if v.miniatura is not None:
            h, w = v.miniatura.shape[:2]
            esc = min(thumb_w / w, thumb_h / h)
            nw, nh = max(1, int(w * esc)), max(1, int(h * esc))
            im = cv2.resize(v.miniatura, (nw, nh), interpolation=cv2.INTER_AREA)
            oy, ox = (thumb_h - nh) // 2, (thumb_w - nw) // 2
            hoja[y0 + oy:y0 + oy + nh, x0 + ox:x0 + ox + nw] = im
        else:
            cv2.putText(hoja, "sin imagen", (x0 + 10, y0 + thumb_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1, cv2.LINE_AA)
        g = f"G{v.grupo}" if v.grupo else "-"
        cv2.putText(hoja, f"{v.id}  {g}  {fmt_duration(v.duracion_s)}"[:40], (x0 + 6, y0 + thumb_h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(hoja, v.nombre[:44], (x0 + 6, y0 + thumb_h + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (190, 190, 190), 1, cv2.LINE_AA)
        cv2.rectangle(hoja, (x0, y0), (x0 + thumb_w - 1, y0 + thumb_h + banda - 1), (60, 60, 60), 1)
    return hoja


CSV_COLS = ["id", "nombre", "ruta", "tam_bytes", "tam_gb", "duracion_s", "duracion", "fps", "ancho", "alto", "frames", "modificado", "huella", "grupo_visual", "duplicado_de", "error"]


def csv_row(v: Video) -> Dict[str, Any]:
    return {"id": v.id, "nombre": v.nombre, "ruta": v.ruta, "tam_bytes": v.tam_bytes, "tam_gb": f"{v.tam_bytes / 1e9:.3f}",
            "duracion_s": "" if v.duracion_s is None else f"{v.duracion_s:.1f}", "duracion": fmt_duration(v.duracion_s) if v.duracion_s is not None else "",
            "fps": "" if v.fps is None else f"{v.fps:.2f}", "ancho": v.ancho or "", "alto": v.alto or "", "frames": v.frames or "",
            "modificado": v.modificado, "huella": v.huella[:12], "grupo_visual": f"G{v.grupo}" if v.grupo else "",
            "duplicado_de": v.duplicado_de, "error": v.error}


def inventory(
    carpetas: Sequence[str],
    salida: str,
    *,
    sin_abrir: bool = False,
    umbral: float = 0.9,
    miniaturas: bool = True,
    out: Callable[[str], None] = print,
) -> int:
    if not 0.0 < umbral <= 1.0:
        out("--umbral debe estar entre 0 y 1.")
        return 2
    for c in carpetas:
        if not os.path.exists(c):
            out(f"No existe: {c}")
            return 2
    rutas = scan(carpetas)
    if not rutas:
        out("No se encontró ningún video (extensiones: " + ", ".join(VIDEO_EXTS) + ").")
        return 2
    if not sin_abrir and any(looks_like_drive(c) for c in carpetas):
        out("AVISO: esta carpeta parece estar en Google Drive. Abrir un video que solo existe en la nube puede descargarlo "
            "entero (~1 GB cada uno). Si no estás seguro de que estén en tu disco, cancela (Ctrl+C) y usa --sin-abrir.")
    ancho_id = max(2, len(str(len(rutas))))
    videos: List[Video] = []
    for i, p in enumerate(rutas, 1):
        try:
            st = os.stat(p)
            tam, mod = st.st_size, datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
        except OSError as exc:
            videos.append(Video(f"V{i:0{ancho_id}d}", p, 0, "", error=f"no se pudo leer el archivo: {exc}"))
            continue
        videos.append(Video(f"V{i:0{ancho_id}d}", p, tam, mod))
    if not sin_abrir:
        for k, v in enumerate(videos, 1):
            if not v.error:
                out(f"  leyendo {k}/{len(videos)}: {v.nombre}")
                read_video(v)
        mark_duplicates(videos)
        grupos, matriz = group_by_similarity([v.firma for v in videos], umbral)
        for v, g in zip(videos, grupos):
            v.grupo = g
    os.makedirs(salida, exist_ok=True)
    with open(os.path.join(salida, "inventario.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(csv_row(v) for v in videos)
    if not sin_abrir:
        with open(os.path.join(salida, "similitud.csv"), "w", newline="", encoding="utf-8-sig") as fh:
            w2 = csv.writer(fh)
            w2.writerow([""] + [v.id for v in videos])
            for i, v in enumerate(videos):
                w2.writerow([v.id] + ["" if np.isnan(matriz[i, j]) else f"{matriz[i, j]:.3f}" for j in range(len(videos))])
        if miniaturas:
            hoja = contact_sheet(videos)
            if hoja is not None and not cv2.imwrite(os.path.join(salida, "hoja.png"), hoja):
                out("AVISO: no se pudo escribir hoja.png")

    # tabla
    out(f"\n{'id':<5} {'nombre':<32} {'GB':>5} {'duración':>9} {'fps':>6} {'resolución':>10} {'grupo':>6}  notas")
    for v in videos:
        notas = []
        if v.duplicado_de:
            notas.append(f"duplicado de {v.duplicado_de}")
        if v.error:
            notas.append(v.error)
        res = f"{v.ancho}x{v.alto}" if v.ancho and v.alto else "?"
        out(f"{v.id:<5} {v.nombre[:32]:<32} {v.tam_bytes / 1e9:>5.2f} {fmt_duration(v.duracion_s):>9} {('%.1f' % v.fps) if v.fps else '?':>6} {res:>10} {('G%d' % v.grupo) if v.grupo else '-':>6}  {'; '.join(notas)}")
    total_gb = sum(v.tam_bytes for v in videos) / 1e9
    out(f"\n{len(videos)} videos, {total_gb:.1f} GB en total.")
    if not sin_abrir:
        n_grupos = len({v.grupo for v in videos if v.grupo})
        out(f"{n_grupos} grupos visuales (mismo fondo = misma arena y cámara, con umbral {umbral}):")
        for g in sorted({v.grupo for v in videos if v.grupo}):
            miembros = [v.id for v in videos if v.grupo == g]
            out(f"  G{g}: {len(miembros)} videos ({', '.join(miembros)})")
        if any(v.duplicado_de for v in videos):
            out("AVISO: hay videos duplicados; usa solo uno de cada pareja.")
        res_distintas = {(v.ancho, v.alto, round(v.fps or 0)) for v in videos if v.abierto and v.ancho}
        if len(res_distintas) > 1:
            out(f"AVISO: hay {len(res_distintas)} combinaciones distintas de resolución y fps: {sorted(res_distintas, key=str)}.")
        out("Reparte los roles por GRUPO, no por video: los videos de un mismo grupo deben ir al mismo rol.")
        out("Si ves videos de la misma arena en grupos distintos (o arenas distintas juntas), ajusta --umbral; mira hoja.png y similitud.csv.")
    out(f"\nInventario en {salida}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Inventario de videos pesados: metadatos, duplicados y grupos por arena/cámara.")
    ap.add_argument("carpetas", nargs="+", help="carpetas (o videos) a inventariar")
    ap.add_argument("--salida", default="inventario", help="carpeta donde escribir (por omisión: inventario)")
    ap.add_argument("--sin-abrir", action="store_true", help="no abrir ningún video: solo nombre, tamaño y fecha (úsalo en carpetas de Drive)")
    ap.add_argument("--umbral", type=float, default=0.9, help="correlación mínima entre fondos para agruparlos (por omisión 0.9)")
    ap.add_argument("--sin-miniaturas", action="store_true", help="no generar hoja.png")
    a = ap.parse_args(argv)
    return inventory(a.carpetas, a.salida, sin_abrir=a.sin_abrir, umbral=a.umbral, miniaturas=not a.sin_miniaturas)


if __name__ == "__main__":
    sys.exit(main())