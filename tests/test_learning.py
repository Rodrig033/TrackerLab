import csv
import importlib
import json
import math
import os
import random
import tempfile
import unittest

import cv2
import numpy as np

from analysis import FrameRecord
from learning.dataset import (
    DatasetCollector,
    make_audit_sheet,
    parse_yolo_label_line,
    split_for_time,
    video_fingerprint,
    yolo_label_line,
)
from learning.metrics import center_distance, evaluate_single_object, iou
from pipeline import FrameEvent, Quit, TrackingJob, run_tracking
from tracking import Detection
from tracking.background_tracker import DEFAULT_PARAMS, BackgroundTracker, compute_background
from video.frame_selection import FrameRange, compute_timing, read_background_frames
from video.reader import VideoReader


def _imp(name):
    try:
        return importlib.import_module(f"tests.{name}")
    except ImportError:
        return importlib.import_module(name)


BG = _imp("test_background_tracker")

FRAME_W, FRAME_H = 400, 300
ROI = (10, 10, 300, 200)  # x0, y0, w0, h0


def evento(i, bbox=(100, 80, 20, 20), status="detectado", t=None, found=True):
    """FrameEvent sintético: un frame con un rectángulo claro donde está el sujeto (coordenadas del recorte)."""
    t = i * 0.1 if t is None else t
    frame = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    x0, y0, _, _ = ROI
    if found:
        bx, by, bw, bh = bbox
        cv2.rectangle(frame, (bx + x0, by + y0), (bx + x0 + bw - 1, by + y0 + bh - 1), (255, 255, 255), -1)
        det = Detection(center=(bx + bw / 2.0, by + bh / 2.0), bbox=bbox)
        rec = FrameRecord(frame=i, time_s=t, dist_acum_px=0.0, cx_abs=1.0, cy_abs_inv=1.0, status=status)
    else:
        det = Detection()
        rec = FrameRecord(frame=i, time_s=t, dist_acum_px=0.0, status=status)
    frame.flags.writeable = False
    return FrameEvent(i, frame, ROI, det, rec, i + 1, 1000)


def coleccionista(tmp, **kw):
    kw.setdefault("min_stable_frames", 3)
    kw.setdefault("stride", 1)
    kw.setdefault("block_seconds", 1000.0)  # todo a entrenamiento salvo que la prueba pida otra cosa
    return DatasetCollector(tmp, video_name="v", **kw)


def archivos(tmp, kind, split):
    d = os.path.join(tmp, kind, split)
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


class FormatoYoloTests(unittest.TestCase):
    def test_linea_conocida(self):
        self.assertEqual(yolo_label_line((10, 20, 30, 40), 100, 200), "0 0.250000 0.200000 0.300000 0.200000")

    def test_ida_y_vuelta(self):
        rng = random.Random(0)
        for _ in range(200):
            w, h = rng.randint(50, 2000), rng.randint(50, 1500)
            bw, bh = rng.uniform(2, w / 2), rng.uniform(2, h / 2)
            box = (rng.uniform(0, w - bw), rng.uniform(0, h - bh), bw, bh)
            cls, back = parse_yolo_label_line(yolo_label_line(box, w, h), w, h)
            self.assertEqual(cls, 0)
            for a, b in zip(box, back):
                self.assertAlmostEqual(a, b, delta=2e-6 * max(w, h))

    def test_se_recorta_al_frame(self):
        _, b = parse_yolo_label_line(yolo_label_line((-10, -10, 30, 30), 100, 100), 100, 100)
        self.assertEqual(tuple(round(v, 4) for v in b), (0.0, 0.0, 20.0, 20.0))
        _, b = parse_yolo_label_line(yolo_label_line((90, 90, 30, 30), 100, 100), 100, 100)
        self.assertEqual(tuple(round(v, 4) for v in b), (90.0, 90.0, 10.0, 10.0))

    def test_cajas_fuera_del_frame_o_vacias_son_error(self):
        for box in ((200, 200, 10, 10), (-50, 10, 20, 10), (10, 10, 0, 5)):
            with self.assertRaises(ValueError, msg=str(box)):
                yolo_label_line(box, 100, 100)
        with self.assertRaises(ValueError):
            parse_yolo_label_line("0 0.5 0.5 0.1", 100, 100)


class RepartoTests(unittest.TestCase):
    def test_ningun_ejemplo_de_entrenamiento_queda_pegado_a_uno_de_validacion(self):
        import bisect
        for buffer_s in (0.5, 1.0, 3.0):
            tiempos = [i * 0.05 for i in range(int(600 / 0.05))]
            train = [t for t in tiempos if split_for_time(t, 20.0, 5, buffer_s) == "train"]
            val = [t for t in tiempos if split_for_time(t, 20.0, 5, buffer_s) == "val"]
            self.assertTrue(train and val)
            separacion = float("inf")
            for v in val:  # entrenamiento más cercano a cada instante de validación
                k = bisect.bisect_left(train, v)
                vecinos = [train[j] for j in (k - 1, k) if 0 <= j < len(train)]
                separacion = min(separacion, min(abs(v - t) for t in vecinos))
            self.assertGreaterEqual(separacion, buffer_s - 1e-9, msg=f"buffer={buffer_s}")

    def test_proporciones_y_bloques(self):
        self.assertEqual(split_for_time(5.0, 20.0, 5, 1.0), "train")  # el primer bloque es de entrenamiento
        self.assertEqual(split_for_time(90.0, 20.0, 5, 1.0), "val")  # bloque 4 (80 a 100 s)
        self.assertIsNone(split_for_time(80.5, 20.0, 5, 1.0))  # inicio del bloque de validación: separación
        self.assertIsNone(split_for_time(79.5, 20.0, 5, 1.0))  # final del bloque de entrenamiento previo: separación
        tiempos = [i * 0.1 for i in range(6000)]
        sp = [split_for_time(t, 20.0, 5, 1.0) for t in tiempos]
        usados = [s for s in sp if s]
        self.assertAlmostEqual(usados.count("val") / len(usados), 0.2, delta=0.04)

    def test_parametros_invalidos(self):
        for args in ((1.0, 1, 0.1), (1.0, 5, 0.6), (0.0, 5, 0.1), (20.0, 5, -1.0)):
            with self.assertRaises(ValueError, msg=str(args)):
                split_for_time(1.0, *args)


class ReglasDelRecolectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_solo_guarda_frames_detectados_y_siempre_devuelve_none(self):
        c = coleccionista(self.tmp.name)
        res = [c(evento(i)) for i in range(5)]
        self.assertEqual(res, [None] * 5)
        for estado, found in (("recuperado", True), ("rechazado", False), ("perdido", False)):
            n = len(archivos(self.tmp.name, "images", "train"))
            c(evento(100, status=estado, found=found))
            self.assertEqual(len(archivos(self.tmp.name, "images", "train")), n, msg=estado)

    def test_estabilidad_minima_y_reinicio_por_un_frame_no_detectado(self):
        c = coleccionista(self.tmp.name, min_stable_frames=3)
        for i in range(5):
            c(evento(i))
        self.assertEqual(len(archivos(self.tmp.name, "images", "train")), 3)  # frames 2, 3 y 4
        c(evento(5, status="rechazado", found=False))
        for i in range(6, 9):
            c(evento(i))
        self.assertEqual(len(archivos(self.tmp.name, "images", "train")), 4)  # tras el rechazo, de nuevo hay que esperar 3
        self.assertEqual(c._skipped["inestable"], 4)  # 2 al inicio y 2 tras el rechazo

    def test_area_atipica_se_descarta_y_no_contamina_la_referencia(self):
        c = coleccionista(self.tmp.name, min_stable_frames=1)
        for i in range(12):
            c(evento(i, bbox=(100, 80, 20, 20)))  # área 400
        c(evento(20, bbox=(50, 50, 80, 60)))  # área 4800: demasiado grande
        c(evento(21, bbox=(100, 80, 22, 20)))  # área 440: normal
        self.assertEqual(c._skipped["area"], 1)
        self.assertEqual(len(archivos(self.tmp.name, "images", "train")), 13)

    def test_caja_que_toca_el_borde_del_roi(self):
        c = coleccionista(self.tmp.name, min_stable_frames=1, border_px=2)
        c(evento(0, bbox=(1, 80, 20, 20)))  # toca el borde izquierdo
        c(evento(1, bbox=(100, 180, 20, 20)))  # pegada abajo: y + h = 200 = alto del ROI
        c(evento(2, bbox=(280, 80, 20, 20)))  # x + w = 300 = ancho del ROI
        c(evento(3, bbox=(100, 80, 20, 20)))  # bien dentro
        self.assertEqual(c._skipped["borde"], 3)
        self.assertEqual(len(archivos(self.tmp.name, "images", "train")), 1)

    def test_muestreo(self):
        c = coleccionista(self.tmp.name, min_stable_frames=1, stride=3)
        for i in range(12):
            c(evento(i))
        self.assertEqual(len(archivos(self.tmp.name, "images", "train")), 4)  # elegibles 0, 3, 6, 9
        self.assertTrue(archivos(self.tmp.name, "images", "train")[0].endswith("0000000.jpg"))
        self.assertEqual(c._skipped["muestreo"], 8)

    def test_reparto_por_tiempo_y_zona_de_separacion(self):
        c = coleccionista(self.tmp.name, min_stable_frames=1, block_seconds=10.0, val_every=3, buffer_s=1.0)
        # bloques de 10 s: 0 y 1 son de entrenamiento; el 2 (20 a 30 s) es de validación; 1 s de separación a cada lado
        for i, t in enumerate([2.0, 5.0, 10.5, 15.0, 19.5, 20.5, 25.0, 29.5]):
            c(evento(i, t=t))
        self.assertEqual(archivos(self.tmp.name, "images", "train"), [f"v_{i:07d}.jpg" for i in range(4)])  # 2, 5, 10.5 y 15 s
        self.assertEqual(archivos(self.tmp.name, "images", "val"), ["v_0000006.jpg"])  # 25 s
        # 19.5 s (entrenamiento pegado a la validación), 20.5 s (inicio de validación) y 29.5 s (final) se descartan
        self.assertEqual(c._skipped["separacion"], 3)

    def test_limite_de_imagenes(self):
        c = coleccionista(self.tmp.name, min_stable_frames=1, max_images=3)
        for i in range(10):
            c(evento(i))
        self.assertEqual(len(archivos(self.tmp.name, "images", "train")), 3)
        self.assertEqual(c._skipped["limite"], 7)

    def test_parametros_invalidos(self):
        for kw in (dict(stride=0), dict(min_stable_frames=0), dict(val_every=1), dict(buffer_s=600.0)):
            with self.assertRaises(ValueError, msg=str(kw)):
                DatasetCollector(os.path.join(self.tmp.name, "x"), **kw)


class AlineacionImagenEtiquetaTests(unittest.TestCase):
    def test_la_caja_guardada_cae_sobre_el_sujeto_con_el_desplazamiento_del_roi(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = coleccionista(tmp, min_stable_frames=1)
            posiciones = [(100, 80, 20, 20), (150, 60, 30, 16), (40, 120, 14, 40)]
            for i, b in enumerate(posiciones):
                c(evento(i, bbox=b))
            for i, b in enumerate(posiciones):
                nombre = f"v_{i:07d}"
                img = cv2.imread(os.path.join(tmp, "images", "train", nombre + ".jpg"), cv2.IMREAD_GRAYSCALE)
                self.assertEqual(img.shape, (FRAME_H, FRAME_W))
                ys, xs = np.nonzero(img > 200)  # píxeles del rectángulo claro
                with open(os.path.join(tmp, "labels", "train", nombre + ".txt")) as f:
                    cls, (x, y, w, h) = parse_yolo_label_line(f.readline(), FRAME_W, FRAME_H)
                self.assertAlmostEqual(x + w / 2, xs.mean() + 0.5, delta=1.5)  # centro de la caja = centro del sujeto
                self.assertAlmostEqual(y + h / 2, ys.mean() + 0.5, delta=1.5)
                self.assertAlmostEqual(w, xs.max() - xs.min() + 1, delta=2.0)
                # y está en coordenadas del FRAME COMPLETO: ROI (10, 10) + posición en el recorte
                self.assertAlmostEqual(x, b[0] + ROI[0], delta=1.0)
                self.assertAlmostEqual(y, b[1] + ROI[1], delta=1.0)


class SalidasTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def generar(self, **kw):
        video = os.path.join(self.tmp.name, "v.bin")
        with open(video, "wb") as f:
            f.write(os.urandom(3 << 20))
        out = os.path.join(self.tmp.name, "ds")
        c = DatasetCollector(out, video_name="rata01", min_stable_frames=1, stride=1, block_seconds=10.0, val_every=3,
                             buffer_s=0.5, tracker_mode="silueta", video_path=video, **kw)
        for i in range(60):
            c(evento(i, t=i * 0.5))
        return out, video, c.finalize()

    def test_data_yaml(self):
        out, _, _ = self.generar()
        with open(os.path.join(out, "data.yaml"), encoding="utf-8") as f:
            texto = f.read()
        self.assertIn(f"path: {json.dumps(os.path.abspath(out))}", texto)
        self.assertIn("train: images/train", texto)
        self.assertIn("val: images/val", texto)
        self.assertIn("names:\n  0: mouse", texto)

    def test_el_manifiesto_coincide_con_los_archivos(self):
        out, _, info = self.generar()
        with open(os.path.join(out, "manifest.csv"), newline="", encoding="utf-8") as f:
            filas = list(csv.DictReader(f))
        self.assertEqual(len(filas), info["imagenes"]["total"])
        for fila in filas:
            split, nombre = fila["archivo"].split("/")
            self.assertTrue(os.path.exists(os.path.join(out, "images", split, nombre + ".jpg")))
            self.assertTrue(os.path.exists(os.path.join(out, "labels", split, nombre + ".txt")))
        for split in ("train", "val"):
            self.assertEqual(len(archivos(out, "images", split)), len(archivos(out, "labels", split)))
            self.assertEqual(len(archivos(out, "images", split)), info["imagenes"][split])
        self.assertGreater(info["imagenes"]["val"], 0)

    def test_procedencia_y_advertencias(self):
        out, video, info = self.generar()
        with open(os.path.join(out, "dataset_info.json"), encoding="utf-8") as f:
            leido = json.load(f)
        self.assertEqual(leido, info)
        self.assertEqual(info["origen_de_las_etiquetas"], "seguimiento_clasico")
        self.assertIn("AUTOMÁTICAS", info["aviso"])
        self.assertIn("circular", info["aviso"])
        self.assertEqual(info["modo_de_seguimiento"], "silueta")
        self.assertEqual(info["parametros"]["buffer_s"], 0.5)
        self.assertEqual(info["video"]["nombre"], "rata01")
        self.assertEqual(info["video"]["size_bytes"], os.path.getsize(video))
        self.assertIn("sha256_extremos", info["video"])
        self.assertEqual(info["clases"], ["mouse"])
        self.assertIn("separacion", info["frames_descartados_por_motivo"])

    def test_la_huella_del_video(self):
        _, video, _ = self.generar()
        a = video_fingerprint(video)
        self.assertEqual(a, video_fingerprint(video))  # estable
        with open(video, "r+b") as f:
            f.seek(-100, os.SEEK_END)
            f.write(b"x" * 100)  # cambia el final
        self.assertNotEqual(a["sha256_extremos"], video_fingerprint(video)["sha256_extremos"])
        pequeno = os.path.join(self.tmp.name, "p.bin")
        with open(pequeno, "wb") as f:
            f.write(b"hola")
        self.assertEqual(video_fingerprint(pequeno)["size_bytes"], 4)

    def test_sin_video_no_hay_huella_pero_si_el_nombre(self):
        c = DatasetCollector(os.path.join(self.tmp.name, "ds2"), video_name="x")
        info = c.finalize()
        self.assertEqual(info["video"], {"nombre": "x"})
        self.assertEqual(info["imagenes"], {"train": 0, "val": 0, "total": 0})


class HojaDeRevisionTests(unittest.TestCase):
    def test_conjunto_vacio(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(make_audit_sheet(tmp, os.path.join(tmp, "a.png")))

    def test_hoja_con_cajas_y_determinista(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = coleccionista(tmp, min_stable_frames=1, block_seconds=5.0, val_every=3, buffer_s=0.2)
            for i in range(40):
                c(evento(i, t=i * 0.25, bbox=(60 + 3 * i, 80, 24, 24)))
            n = c.finalize()["imagenes"]["total"]
            a = make_audit_sheet(tmp, os.path.join(tmp, "hoja1.png"), n=12, cols=4, thumb_w=200, seed=3)
            b = make_audit_sheet(tmp, os.path.join(tmp, "hoja2.png"), n=12, cols=4, thumb_w=200, seed=3)
            self.assertGreaterEqual(n, 12)
            img = cv2.imread(a)
            self.assertEqual(img.shape[1], 4 * 200)
            self.assertEqual(img.shape[0] % 3, 0)  # 12 ejemplos en 4 columnas = 3 filas de la misma altura
            with open(a, "rb") as fa, open(b, "rb") as fb:
                self.assertEqual(fa.read(), fb.read())  # misma semilla, misma hoja
            b, g, r = (img[:, :, k].astype(int) for k in range(3))
            verde = (g > 120) & (g > b + 60) & (g > r + 60)
            self.assertGreater(int(verde.sum()), 150)  # las cajas se dibujaron


class ConElPipelineRealTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, "v.avi")
        frames, cls.truth = BG.make_frames(90, lost={30, 31, 32})
        vw = cv2.VideoWriter(cls.path, cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (BG.FW, BG.FH))
        for f in frames:
            vw.write(f)
        vw.release()
        cls.mask = BG.roi_mask("elipse")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def correr(self, callback, **kw):
        with VideoReader(self.path) as r:
            bg = compute_background(read_background_frames(r, 35), BG.ROI, self.mask, False)
            job = TrackingJob(r, FrameRange(0, 85), compute_timing(r.fps), BackgroundTracker(self.mask, bg, DEFAULT_PARAMS),
                              BG.ROI, start=((40.0, 90.0), (28, 78, 24, 24)))
            return run_tracking(job, on_frame=callback, keep_records=True, log=None)

    def test_las_etiquetas_caen_sobre_el_raton_real_y_el_reparto_es_disjunto(self):
        out = os.path.join(self.tmp.name, "ds")
        c = DatasetCollector(out, video_name="sint", min_stable_frames=5, stride=3, block_seconds=1.0, val_every=3, buffer_s=0.1)
        self.correr(c)
        info = c.finalize()
        self.assertGreater(info["imagenes"]["train"], 5)
        self.assertGreater(info["imagenes"]["val"], 0)
        with open(os.path.join(out, "manifest.csv"), newline="", encoding="utf-8") as f:
            filas = list(csv.DictReader(f))
        for fila in filas:
            idx = int(fila["frame"])
            self.assertNotIn(idx, (30, 31, 32))  # los frames sin sujeto no se usan
            tx, ty = self.truth[idx]
            cx, cy = float(fila["x"]) + float(fila["w"]) / 2, float(fila["y"]) + float(fila["h"]) / 2
            self.assertLess(math.dist((cx, cy), (tx, ty)), 3.0, msg=f"frame {idx}")  # la caja está sobre el ratón real
        t_train = sorted(float(f["time_s"]) for f in filas if f["split"] == "train")
        t_val = sorted(float(f["time_s"]) for f in filas if f["split"] == "val")
        self.assertGreaterEqual(min(abs(a - b) for a in t_train for b in t_val), 0.1 - 1e-9)

    def test_coleccionar_no_altera_el_seguimiento(self):
        sin = self.correr(None)
        c = DatasetCollector(os.path.join(self.tmp.name, "ds2"), min_stable_frames=2, stride=2)
        con = self.correr(c)
        self.assertEqual(sin.records, con.records)  # idénticos, registro por registro

    def test_es_compatible_con_las_ordenes_de_la_interfaz(self):
        c = DatasetCollector(os.path.join(self.tmp.name, "ds3"), min_stable_frames=2, stride=1)
        res = self.correr(lambda ev: (c(ev), Quit() if ev.processed == 12 else None)[1])
        self.assertTrue(res.stopped_by_user)
        self.assertEqual(res.frames_processed, 12)


class MetricasTests(unittest.TestCase):
    def test_iou(self):
        self.assertEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 5, 5)), 0.0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 10, 10)), 50 / 150)
        self.assertEqual(iou((0, 0, 10, 10), (10, 0, 10, 10)), 0.0)  # solo se tocan en el borde
        self.assertAlmostEqual(iou((0, 0, 10, 10), (2, 2, 4, 4)), 16 / 100)  # una dentro de la otra
        self.assertEqual(iou((0, 0, 0, 0), (0, 0, 0, 0)), 0.0)

    def test_iou_es_simetrico(self):
        rng = random.Random(1)
        for _ in range(100):
            a = (rng.uniform(0, 50), rng.uniform(0, 50), rng.uniform(1, 40), rng.uniform(1, 40))
            b = (rng.uniform(0, 50), rng.uniform(0, 50), rng.uniform(1, 40), rng.uniform(1, 40))
            self.assertAlmostEqual(iou(a, b), iou(b, a))
            self.assertTrue(0.0 <= iou(a, b) <= 1.0)

    def test_distancia_entre_centros(self):
        self.assertEqual(center_distance((0, 0, 10, 10), (30, 40, 10, 10)), 50.0)

    def test_evaluacion_cuenta_cada_caso(self):
        B = (0, 0, 10, 10)
        preds = [B, B, None, (50, 50, 10, 10), B, None]
        truth = [B, (2, 0, 10, 10), B, B, None, None]
        r = evaluate_single_object(preds, truth)
        # 0: acierto | 1: acierto (IoU 0.67) | 2: perdido | 3: lugar equivocado | 4: falso positivo | 5: ambos ausentes
        self.assertEqual((r.frames, r.with_truth, r.true_positive, r.wrong_place, r.missed, r.false_positive), (6, 4, 2, 1, 1, 1))
        self.assertAlmostEqual(r.recall, 2 / 4)
        self.assertAlmostEqual(r.precision, 2 / 4)  # dio 4 cajas (2 buenas, 1 mal ubicada, 1 falsa)
        self.assertAlmostEqual(r.mean_iou, (1.0 + 80 / 120) / 2)
        self.assertAlmostEqual(r.mean_center_error, (0.0 + 2.0) / 2)

    def test_umbral_de_iou(self):
        B, P = (0, 0, 10, 10), (3, 0, 10, 10)  # IoU = 70/130 = 0.54
        self.assertEqual(evaluate_single_object([P], [B], 0.5).true_positive, 1)
        self.assertEqual(evaluate_single_object([P], [B], 0.75).wrong_place, 1)

    def test_casos_limite(self):
        r = evaluate_single_object([None, None], [None, None])
        self.assertEqual((r.recall, r.precision, r.mean_iou, r.mean_center_error), (None, None, None, None))
        with self.assertRaises(ValueError):
            evaluate_single_object([None], [None, None])

    def test_comparar_el_seguimiento_con_una_referencia(self):
        """Uso previsto: ¿cuánto coincide una detección con la referencia, frame a frame?"""
        ref = [(10 + i, 20, 12, 12) for i in range(50)]
        pred = [b if i % 10 else None for i, b in enumerate(ref)]  # se pierde 1 de cada 10
        r = evaluate_single_object(pred, ref)
        self.assertEqual((r.true_positive, r.missed, r.false_positive), (45, 5, 0))
        self.assertAlmostEqual(r.recall, 0.9)
        self.assertEqual(r.precision, 1.0)


if __name__ == "__main__":
    unittest.main()