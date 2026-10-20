import ast
import csv
import importlib
import json
import math
import os
import tempfile
import unittest
from datetime import datetime

import cv2
import numpy as np

from learning.dataset import DatasetCollector
from learning.session import DatasetSession
from pipeline import Quit, TrackingJob, run_tracking
from tracking.background_tracker import DEFAULT_PARAMS, BackgroundTracker, compute_background
from video.frame_selection import FrameRange, compute_timing, read_background_frames
from video.reader import VideoReader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _imp(name):
    try:
        return importlib.import_module(f"tests.{name}")
    except ImportError:
        return importlib.import_module(name)


TL = _imp("test_learning")
TP = _imp("test_pipeline")

AHORA = lambda: datetime(2026, 10, 20, 15, 30, 45)  # noqa: E731


def sesion(tmp, enabled=True, **kw):
    mensajes = []
    kw.setdefault("log", mensajes.append)
    kw.setdefault("now", AHORA)
    return DatasetSession(enabled, os.path.join(tmp, "exp"), **kw), mensajes


class ApagadoTests(unittest.TestCase):
    def test_wrap_devuelve_el_mismo_on_frame_sin_envolverlo(self):
        with tempfile.TemporaryDirectory() as tmp:
            s, _ = sesion(tmp, enabled=False)
            f = lambda ev: None  # noqa: E731
            self.assertIs(s.wrap(f), f)
            self.assertIsNone(s.wrap(None))

    def test_no_crea_carpetas_no_escribe_ni_avisa(self):
        with tempfile.TemporaryDirectory() as tmp:
            s, mensajes = sesion(tmp, enabled=False, video_path="x.mp4", duration_s=60.0)
            self.assertIsNone(s.finish())
            self.assertIsNone(s.finish(TP.run_script([100, 110])))
            self.assertFalse(os.path.exists(os.path.join(tmp, "exp")))
            self.assertEqual(mensajes, [])
            self.assertIsNone(s.collector)

    def test_el_seguimiento_es_identico_con_una_sesion_apagada(self):
        sin = TP.run_script([100, 110, 500, 120, None, 130])
        reader = TP.FakeReader()
        with tempfile.TemporaryDirectory() as tmp:
            s, _ = sesion(tmp, enabled=False)
            job = TrackingJob(reader, FrameRange(0, 5), compute_timing(30.0), TP.Scripted([100, 110, 500, 120, None, 130]),
                              (0, 0, 4000, 200), start=((100.0, 100.0), (95, 95, 10, 10)))
            con = run_tracking(job, on_frame=s.wrap(None), keep_records=True, log=None)
        self.assertEqual(sin.records, con.records)

class EncendidoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_crea_la_carpeta_con_fecha_dentro_del_experimento(self):
        s, _ = sesion(self.tmp.name, video_path="/videos/Rata 01.mp4")
        esperada = os.path.join(self.tmp.name, "exp", "dataset_20261020_153045")
        self.assertEqual(s.out_dir, esperada)
        for sub in ("images/train", "images/val", "labels/train", "labels/val"):
            self.assertTrue(os.path.isdir(os.path.join(esperada, sub)), msg=sub)
        self.assertEqual(s.collector.video_name, "Rata_01")  # nombre seguro para archivos

    def test_el_on_frame_de_la_interfaz_va_primero_y_sus_ordenes_pasan_intactas(self):
        s, _ = sesion(self.tmp.name)
        orden = []

        class Espia:
            def __call__(self, ev):
                orden.append("generador")

        s.collector = Espia()
        quit_ = Quit()

        def interfaz(ev):
            orden.append("interfaz")
            return quit_

        cmd = s.wrap(interfaz)("evento")
        self.assertIs(cmd, quit_)  # la orden (tecla 'q') llega intacta a run_tracking
        self.assertEqual(orden, ["interfaz", "generador"])

    def test_sin_on_frame_tambien_alimenta_al_generador(self):
        s, _ = sesion(self.tmp.name)
        vistos = []
        s.collector = lambda ev: vistos.append(ev)
        self.assertIsNone(s.wrap(None)("e"))
        self.assertEqual(vistos, ["e"])

    def test_un_fallo_del_generador_no_detiene_el_seguimiento(self):
        s, mensajes = sesion(self.tmp.name)

        class Falla(DatasetCollector):
            llamadas = 0

            def __call__(self, event):
                Falla.llamadas += 1
                if Falla.llamadas == 2:
                    raise OSError("disco lleno")
                return super().__call__(event)

        s.collector = Falla(s.out_dir, video_name="v", min_stable_frames=1, stride=1, block_seconds=1000.0)
        ordenes = [s.wrap(lambda ev: "orden")(TL.evento(i)) for i in range(6)]
        self.assertEqual(ordenes, ["orden"] * 6)  # el seguimiento sigue recibiendo sus órdenes
        self.assertEqual(Falla.llamadas, 2)  # tras el fallo, el generador ya no se vuelve a llamar
        self.assertIn("OSError: disco lleno", s.error)
        self.assertEqual(sum("se desactivó" in m for m in mensajes), 1)  # se avisa una sola vez
        s.finish()
        self.assertTrue(any("incompleto" in m for m in mensajes))

    def test_si_no_se_puede_preparar_el_generador_el_seguimiento_continua(self):
        archivo = os.path.join(self.tmp.name, "exp")  # un ARCHIVO donde debería ir la carpeta
        open(archivo, "w").close()
        mensajes = []
        s = DatasetSession(True, archivo, log=mensajes.append, now=AHORA)
        f = lambda ev: None  # noqa: E731
        self.assertIsNone(s.collector)
        self.assertIs(s.wrap(f), f)
        self.assertIsNone(s.finish())
        self.assertTrue(any("No se pudo preparar" in m for m in mensajes))

    def test_los_bloques_se_acortan_en_videos_cortos_para_que_exista_validacion(self):
        for duracion, bloque, margen in ((None, 20.0, 1.0), (600.0, 20.0, 1.0), (120.0, 12.0, 1.0), (60.0, 6.0, 1.0),
                                         (30.0, 4.0, 0.8), (5.0, 4.0, 0.8)):
            s, _ = sesion(self.tmp.name, enabled=False, duration_s=duracion)
            self.assertEqual((s.block_seconds, s.buffer_s), (bloque, margen), msg=f"duración {duracion}")
            self.assertLess(2 * s.buffer_s, s.block_seconds)

    def test_finish_es_idempotente(self):
        s, mensajes = sesion(self.tmp.name)
        a = s.finish()
        n = len(mensajes)
        b = s.finish()
        self.assertIs(a, b)
        self.assertEqual(len(mensajes), n)  # no repite el aviso ni reescribe

    def test_avisos_del_cierre(self):
        s, mensajes = sesion(self.tmp.name)
        s.finish()
        self.assertTrue(any("no se guardó ninguna imagen" in m for m in mensajes))  # no entró ningún frame
        s2, m2 = sesion(self.tmp.name, now=lambda: datetime(2026, 10, 20, 15, 31, 0))
        for i in range(20):
            s2.collector(TL.evento(i, t=i * 1.0))
        s2.finish()
        self.assertFalse(any("ninguna imagen" in m for m in m2))
        self.assertTrue(any("no hay imágenes de validación" in m for m in m2))  # 20 s con bloques de 20 s: todo es entrenamiento

def escena_larga(n, w=160, h=120):
    rng = np.random.default_rng(5)
    frames, verdad = [], []
    for i in range(n):
        f = np.clip(np.full((h, w, 3), 100, np.int16) + rng.normal(0, 2, (h, w, 3)).astype(np.int16), 0, 255).astype(np.uint8)
        cx, cy = 80 + 45 * math.sin(i / 37.0), 60 + 25 * math.sin(i / 23.0)
        cv2.circle(f, (int(cx), int(cy)), 9, (230, 230, 230), -1)
        frames.append(f)
        verdad.append((cx, cy))
    return frames, verdad


class RecorridoCompletoTests(unittest.TestCase):
    W, H = 160, 120

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.videos = {}
        for n in (600, 100):  # 24 s y 4 s a 25 fps
            frames, verdad = escena_larga(n)
            path = os.path.join(cls.tmp.name, f"v{n}.avi")
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (cls.W, cls.H))
            for f in frames:
                vw.write(f)
            vw.release()
            cls.videos[n] = (path, verdad)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def correr(self, n, sesion_, quit_en=None):
        path, _ = self.videos[n]
        mask = np.full((self.H, self.W), 255, np.uint8)
        roi = (0, 0, self.W, self.H)
        with VideoReader(path) as r:
            bg = compute_background(read_background_frames(r, 35), roi, mask, False)
            job = TrackingJob(r, FrameRange(0, n - 1), compute_timing(r.fps), BackgroundTracker(mask, bg, DEFAULT_PARAMS), roi,
                              start=((80.0, 60.0), (70, 50, 20, 20)))
            if sesion_ is None:
                cb = None
            else:
                base = (lambda ev: Quit() if quit_en and ev.processed == quit_en else None)
                cb = sesion_.wrap(base)
            res = run_tracking(job, on_frame=cb, keep_records=True, log=None)
        if sesion_ is not None:
            sesion_.finish(res)
        return res

    def test_de_punta_a_punta(self):
        path, verdad = self.videos[600]
        s, mensajes = sesion(self.tmp.name, video_path=path, duration_s=24.0)
        res = self.correr(600, s)
        out = s.out_dir
        for archivo in ("data.yaml", "manifest.csv", "dataset_info.json", "revision.png"):
            self.assertTrue(os.path.exists(os.path.join(out, archivo)), msg=archivo)
        self.assertIsNotNone(cv2.imread(os.path.join(out, "revision.png")))
        with open(os.path.join(out, "dataset_info.json"), encoding="utf-8") as f:
            info = json.load(f)
        self.assertGreaterEqual(info["imagenes"]["train"], 5)
        self.assertGreaterEqual(info["imagenes"]["val"], 1)
        # la ficha incluye cómo se procesó la corrida que generó las etiquetas
        self.assertEqual(info["opciones_de_procesamiento"], res.settings)
        self.assertEqual(info["modo_de_seguimiento"], "fondo")
        self.assertIn("sha256_extremos", info["video"])
        with open(os.path.join(out, "manifest.csv"), newline="", encoding="utf-8") as f:
            filas = sorted(csv.DictReader(f), key=lambda r: float(r["time_s"]))
        tiempos = [float(r["time_s"]) for r in filas]
        self.assertTrue(all(b - a >= 1.0 - 1e-6 for a, b in zip(tiempos, tiempos[1:])))  # a lo sumo una imagen por segundo
        for fila in filas:  # las cajas caen sobre el ratón real
            cx, cy = float(fila["x"]) + float(fila["w"]) / 2, float(fila["y"]) + float(fila["h"]) / 2
            self.assertLess(math.dist((cx, cy), verdad[int(fila["frame"])]), 3.0)
        texto = "\n".join(mensajes)
        self.assertIn(out, texto)
        self.assertIn(os.path.join(out, "revision.png"), texto)
        self.assertIn("AUTOMÁTICAS", texto)
        self.assertNotIn("AVISO", texto)

    def test_el_seguimiento_no_cambia_por_generar_datos(self):
        sin = self.correr(600, None)
        s, _ = sesion(self.tmp.name, now=lambda: datetime(2026, 10, 20, 16, 0, 0), duration_s=24.0)
        con = self.correr(600, s)
        self.assertEqual(sin.records, con.records)  # idéntico, registro por registro

    def test_la_tecla_q_detiene_la_corrida_y_el_conjunto_se_cierra_igual(self):
        s, mensajes = sesion(self.tmp.name, now=lambda: datetime(2026, 10, 20, 16, 1, 0), duration_s=24.0)
        res = self.correr(600, s, quit_en=300)
        self.assertTrue(res.stopped_by_user)
        self.assertEqual(res.frames_processed, 300)
        self.assertTrue(os.path.exists(os.path.join(s.out_dir, "dataset_info.json")))
        with open(os.path.join(s.out_dir, "manifest.csv"), newline="", encoding="utf-8") as f:
            self.assertTrue(all(int(r["frame"]) < 300 for r in csv.DictReader(f)))

    def test_video_corto_avisa_que_no_hay_validacion(self):
        s, mensajes = sesion(self.tmp.name, now=lambda: datetime(2026, 10, 20, 16, 2, 0), duration_s=4.0)
        self.correr(100, s)
        self.assertTrue(any("no hay imágenes de validación" in m for m in mensajes))

    def test_el_limite_de_imagenes_se_avisa(self):
        s, mensajes = sesion(self.tmp.name, now=lambda: datetime(2026, 10, 20, 16, 3, 0), duration_s=24.0, max_images=3)
        self.correr(600, s)
        with open(os.path.join(s.out_dir, "manifest.csv"), newline="", encoding="utf-8") as f:
            self.assertEqual(len(list(csv.DictReader(f))), 3)
        self.assertTrue(any("límite de 3 imágenes" in m for m in mensajes))

class EnlaceEnMainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ruta = os.path.join(ROOT, "main.py")
        if not os.path.exists(ruta):
            raise unittest.SkipTest("no se encontró main.py en la raíz del proyecto")
        with open(ruta, encoding="utf-8") as f:
            cls.tree = ast.parse(f.read())
        cls.main_fn = next((n for n in cls.tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
        if cls.main_fn is None:
            raise unittest.SkipTest("main.py no define main()")

    def nodo(self, tipo, cond):
        return [n for n in ast.walk(self.main_fn) if isinstance(n, tipo) and cond(n)]

    def llamada(self, nombre):
        return self.nodo(ast.Assign, lambda n: isinstance(n.value, ast.Call) and getattr(n.value.func, "id", "") == nombre)

    def test_el_interruptor_existe_y_esta_apagado(self):
        asig = [n for n in self.tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "COLLECT_DATASET" for t in n.targets)]
        self.assertEqual(len(asig), 1, "falta COLLECT_DATASET al inicio de main.py")
        self.assertIsInstance(asig[0].value, ast.Constant)
        self.assertIs(asig[0].value.value, False, "COLLECT_DATASET debe estar en False por defecto")

    def test_se_importa_la_sesion(self):
        imp = [n for n in self.tree.body if isinstance(n, ast.ImportFrom) and n.module == "learning.session"]
        self.assertTrue(imp and any(a.name == "DatasetSession" for a in imp[0].names))

    def test_la_sesion_recibe_el_interruptor_y_los_datos_del_video(self):
        sesion_ = self.nodo(ast.Assign, lambda n: isinstance(n.value, ast.Call) and getattr(n.value.func, "id", "") == "DatasetSession")
        self.assertEqual(len(sesion_), 1, "falta crear la DatasetSession en main()")
        call = sesion_[0].value
        self.assertEqual(ast.unparse(call.args[0]), "COLLECT_DATASET")
        claves = {k.arg: ast.unparse(k.value) for k in call.keywords}
        self.assertEqual(claves["video_path"], "video_path")
        self.assertIn("timing.video_fps", claves["duration_s"])
        self.assertEqual(sesion_[0].targets[0].id, "dataset")

    def test_on_frame_se_envuelve_y_on_record_queda_igual(self):
        run = self.llamada("run_tracking")
        self.assertEqual(len(run), 1)
        kws = {k.arg: ast.unparse(k.value) for k in run[0].value.keywords}
        self.assertEqual(kws["on_frame"], "dataset.wrap(on_frame)")
        self.assertEqual(kws["on_record"], "on_record")
        self.assertEqual(kws["keep_records"], "True")

    def test_orden_crear_correr_cerrar_y_cerrar_antes_de_exportar(self):
        crear = self.llamada("DatasetSession")[0].lineno
        correr = self.llamada("run_tracking")[0].lineno
        cerrar = self.nodo(ast.Expr, lambda n: ast.unparse(n.value) == "dataset.finish(result)")
        self.assertEqual(len(cerrar), 1, "falta dataset.finish(result) después de run_tracking")
        exportar = self.nodo(ast.Call, lambda n: getattr(n.func, "id", "") == "export_tracking_csv")
        self.assertTrue(crear < correr < cerrar[0].lineno < exportar[0].lineno,
                        "debe cerrarse el conjunto ANTES de exportar el CSV: si exportar falla (CSV abierto en Excel), "
                        "el conjunto igual queda completo")

    def test_las_variables_que_usa_existen_antes_de_la_sesion(self):
        crear = self.llamada("DatasetSession")[0].lineno
        for nombre in ("video_path", "frame_range", "timing", "options"):
            definidas = self.nodo(ast.Name, lambda n: n.id == nombre and isinstance(n.ctx, ast.Store) and n.lineno < crear)
            self.assertTrue(definidas, f"{nombre} no está definida antes de crear la sesión")
        importa_dir = [n for n in self.tree.body if isinstance(n, ast.ImportFrom) and any(a.name == "experiment_dir" for a in n.names)]
        self.assertTrue(importa_dir)


if __name__ == "__main__":
    unittest.main()