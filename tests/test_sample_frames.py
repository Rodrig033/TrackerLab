"""Pruebas de tools/sample_frames.py.

Ejecutar desde la raíz del proyecto:
    python -m pytest tests/test_sample_frames.py -v
"""

import importlib
import json
import os
import random
import tempfile
import unittest

import cv2
import numpy as np

from persistence.run_inputs import save_run_inputs
from tools import sample_frames as S
from video.reader import VideoReader


def _imp(name):
    try:
        return importlib.import_module(f"tests.{name}")
    except ImportError:
        return importlib.import_module(name)


RI = _imp("test_run_inputs")

W, H = 64, 48


def video_con_indices(tmp, nombre, n=120, fps=25.0):
    """Video en el que el NÚMERO DE FRAME está codificado en bits blanco/negro (8 parches de 8x8 px).

    Los bits en blanco y negro resisten la compresión del video; un solo nivel de gris no (se pierde ±1).
    """
    path = os.path.join(tmp, nombre)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
    for i in range(n):
        f = np.full((H, W, 3), 128, np.uint8)
        for bit in range(8):
            f[:8, bit * 8:(bit + 1) * 8] = 255 if (i >> bit) & 1 else 0
        vw.write(f)
    vw.release()
    return path


def indice_de(img):
    """Inverso de video_con_indices: lee los 8 bits (umbral en 128)."""
    return sum((1 << bit) for bit in range(8) if img[:8, bit * 8:(bit + 1) * 8].mean() > 128)


def leer(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def silencio(_):
    pass

class PlanDeFramesTests(unittest.TestCase):
    def test_alineado_dentro_del_rango_y_ordenado(self):
        for start, end, step, n in ((0, 99, 1, 10), (5, 205, 4, 12), (3, 50, 3, 7)):
            plan = S.plan_frames(start, end, step, n, random.Random(1))
            self.assertEqual(len(plan), n)
            self.assertEqual(plan, sorted(plan))
            self.assertEqual(len(set(plan)), n)
            for f in plan:
                self.assertTrue(start <= f <= end)
                self.assertEqual((f - start) % step, 0, msg=f"{f} no está en la rejilla {start}+k*{step}")

    def test_uno_por_tramo_calculado_de_forma_independiente(self):
        start, end, step, n = 10, 509, 5, 20
        cand = list(range(start, end + 1, step))  # 100 candidatos -> tramos de 5
        for seed in range(25):
            plan = S.plan_frames(start, end, step, n, random.Random(seed))
            for i, f in enumerate(plan):
                self.assertIn(f, cand[i * 5:(i + 1) * 5], msg=f"semilla {seed}, tramo {i}")

    def test_cubre_todo_el_video(self):
        plan = S.plan_frames(0, 999, 1, 10, random.Random(3))
        self.assertLess(plan[0], 100)  # el primer frame cae en el primer décimo
        self.assertGreaterEqual(plan[-1], 900)  # y el último, en el último
        self.assertLess(max(b - a for a, b in zip(plan, plan[1:])), 200)  # ningún hueco enorme

    def test_si_piden_mas_de_los_que_hay_se_devuelven_todos(self):
        self.assertEqual(S.plan_frames(0, 9, 2, 50, random.Random(0)), [0, 2, 4, 6, 8])
        self.assertEqual(S.plan_frames(0, 9, 1, 10, random.Random(0)), list(range(10)))

    def test_rango_vacio_y_validaciones(self):
        self.assertEqual(S.plan_frames(10, 5, 1, 3, random.Random(0)), [])
        with self.assertRaises(ValueError):
            S.plan_frames(0, 10, 0, 3, random.Random(0))
        with self.assertRaises(ValueError):
            S.plan_frames(0, 10, 1, 0, random.Random(0))

    def test_reproducible_y_la_semilla_cambia_el_resultado(self):
        a = S.plan_frames(0, 999, 1, 15, random.Random(7))
        self.assertEqual(a, S.plan_frames(0, 999, 1, 15, random.Random(7)))
        self.assertNotEqual(a, S.plan_frames(0, 999, 1, 15, random.Random(8)))


class NombresTests(unittest.TestCase):
    def test_safe_stem(self):
        self.assertEqual(S.safe_stem("/x/Mi video (1).mp4"), "Mi_video_1")
        self.assertEqual(S.safe_stem("rata-01_final.avi"), "rata-01_final")
        self.assertEqual(S.safe_stem("???.mp4"), "video")  # nada utilizable -> nombre por omisión
        self.assertRegex(S.safe_stem("Ratón día 3.mov"), r"^[0-9A-Za-z_-]+$")

    def test_image_name(self):
        self.assertEqual(S.image_name("rata01", 42), "rata01_f000042.jpg")
        self.assertEqual(S.image_name("rata01", 1234567, "png"), "rata01_f1234567.png")


class MuestreoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.v1 = video_con_indices(cls.tmp.name, "uno.avi", 120)
        cls.v2 = video_con_indices(cls.tmp.name, "dos.avi", 90)
        os.makedirs(os.path.join(cls.tmp.name, "otra"))
        cls.v1b = video_con_indices(os.path.join(cls.tmp.name, "otra"), "uno.avi", 60)  # mismo nombre, otro video

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def salida(self, nombre):
        return os.path.join(self.tmp.name, "val_" + nombre)

    def correr(self, nombre, videos, **kw):
        kw.setdefault("por_video", 10)
        kw.setdefault("formato", "png")
        kw.setdefault("out", silencio)
        s = self.salida(nombre)
        fuentes = [S.Fuente(v, step=kw.pop("paso", 1), desde="video|0.0|None") for v in videos]
        return S.muestrear(fuentes, s, **kw), s

    def manifiesto(self, s):
        return json.loads(leer(os.path.join(s, "manifest.json")))

    def test_cada_imagen_es_el_frame_que_dice_el_manifiesto(self):
        rc, s = self.correr("contenido", [self.v1, self.v2])
        self.assertEqual(rc, 0)
        man = self.manifiesto(s)
        self.assertEqual(len(man["videos"]), 2)
        for v in man["videos"]:
            self.assertEqual(len(v["frames"]), 10)
            for f in v["frames"]:
                img = cv2.imread(os.path.join(s, "images", S.image_name(v["id"], f, "png")))
                self.assertIsNotNone(img)
                self.assertEqual(indice_de(img), f, msg=f"{v['id']} frame {f}: la imagen guardada es otro frame")

    def test_manifiesto_completo(self):
        _, s = self.correr("manifiesto", [self.v1])
        man = self.manifiesto(s)
        v = man["videos"][0]
        self.assertEqual((man["clases"], man["ciego"], man["proposito"]), (["raton"], True, "validacion"))
        self.assertEqual((v["ancho"], v["alto"], v["total_frames"], v["rango"], v["paso"]), (W, H, 120, [0, 119], 1))
        self.assertEqual(v["fps"], 25.0)
        self.assertEqual(len(v["sha256_extremos"]), 64)
        self.assertTrue(os.path.isabs(v["path"]))

    def test_es_ciego_no_hay_etiquetas_sugeridas_y_se_escriben_las_guias(self):
        _, s = self.correr("ciego", [self.v1])
        self.assertEqual(os.listdir(os.path.join(s, "labels")), [])
        self.assertEqual(leer(os.path.join(s, "classes.txt")).strip(), "raton")
        txt = leer(os.path.join(s, "LEEME_ETIQUETADO.txt"))
        for clave in ("SIN la cola", "VACÍA", "No uses cajas sugeridas"):
            self.assertIn(clave, txt)

    def test_alineado_a_la_rejilla_del_pipeline(self):
        _, s = self.correr("paso", [self.v1], paso=4, por_video=12)
        for f in self.manifiesto(s)["videos"][0]["frames"]:
            self.assertEqual(f % 4, 0)

    def test_con_entradas_usa_su_rango_y_su_frame_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            RI.Escenas.preparar(tmp)
            with VideoReader(RI.Escenas.p_bg) as r:
                job = RI.Escenas.job_fondo(r)  # FrameRange(0, 85) con frame_step 2
                carpeta = save_run_inputs(os.path.join(tmp, "exp"), job, video_path=RI.Escenas.p_bg, options=RI.Escenas.OPT_FONDO,
                                          sample_idx=0, app_version="1.0", log=None, now=RI.AHORA)
            fuentes = S.fuentes_desde_args([], [carpeta], paso=1, desde_s=0.0, hasta_s=None)
            self.assertEqual((fuentes[0].start, fuentes[0].end, fuentes[0].step), (0, 85, 2))
            s = os.path.join(tmp, "val")
            self.assertEqual(S.muestrear(fuentes, s, por_video=15, out=silencio), 0)
            v = json.loads(leer(os.path.join(s, "manifest.json")))["videos"][0]
            self.assertEqual(v["rango"], [0, 85])
            for f in v["frames"]:
                self.assertEqual(f % 2, 0)  # todos son frames que el pipeline procesa
                self.assertLessEqual(f, 85)

    def test_limites_en_segundos(self):
        fuentes = S.fuentes_desde_args([self.v1], [], paso=1, desde_s=1.0, hasta_s=2.0)  # 25 fps -> frames 25..50
        s = self.salida("segundos")
        self.assertEqual(S.muestrear(fuentes, s, por_video=8, formato="png", out=silencio), 0)
        for f in self.manifiesto(s)["videos"][0]["frames"]:
            self.assertTrue(25 <= f <= 50, msg=str(f))

    def test_reproducible_y_agregar_videos_no_cambia_los_de_los_demas(self):
        _, a = self.correr("rep_a", [self.v1])
        _, b = self.correr("rep_b", [self.v1, self.v2])  # con un video más
        self.assertEqual(self.manifiesto(a)["videos"][0]["frames"], self.manifiesto(b)["videos"][0]["frames"])
        _, c = self.correr("rep_c", [self.v1], semilla=99)
        self.assertNotEqual(self.manifiesto(a)["videos"][0]["frames"], self.manifiesto(c)["videos"][0]["frames"])

    def test_dos_videos_con_el_mismo_nombre_tienen_ids_distintos(self):
        rc, s = self.correr("mismo_nombre", [self.v1, self.v1b])
        self.assertEqual(rc, 0)
        ids = [v["id"] for v in self.manifiesto(s)["videos"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(os.listdir(os.path.join(s, "images"))), 20)  # ninguna imagen pisó a otra

    def test_no_sobrescribe_sin_permiso(self):
        _, s = self.correr("protegido", [self.v1])
        antes = leer(os.path.join(s, "manifest.json"))
        mensajes = []
        rc = S.muestrear([S.Fuente(self.v2, desde="video|0.0|None")], s, por_video=5, formato="png", out=mensajes.append)
        self.assertEqual(rc, 2)
        self.assertTrue(any("--anadir" in m for m in mensajes))
        self.assertEqual(leer(os.path.join(s, "manifest.json")), antes)

    def test_anadir_agrega_sin_tocar_lo_anterior_y_omite_repetidos(self):
        _, s = self.correr("anadir", [self.v1])
        antes = self.manifiesto(s)["videos"][0]
        rc = S.muestrear([S.Fuente(self.v2, desde="video|0.0|None"), S.Fuente(self.v1, desde="video|0.0|None")], s, por_video=10,
                         formato="png", anadir=True, out=silencio)
        self.assertEqual(rc, 0)
        man = self.manifiesto(s)
        self.assertEqual([v["nombre"] for v in man["videos"]], ["uno.avi", "dos.avi"])  # uno.avi no se repite
        self.assertEqual(man["videos"][0], antes)

    def test_sobrescribir_no_borra_las_etiquetas(self):
        _, s = self.correr("etiquetas", [self.v1])
        etiqueta = os.path.join(s, "labels", "uno_f000001.txt")
        with open(etiqueta, "w") as fh:
            fh.write("0 0.5 0.5 0.2 0.2\n")
        rc = S.muestrear([S.Fuente(self.v1, desde="video|0.0|None")], s, por_video=10, formato="png", sobrescribir=True, out=silencio)
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(etiqueta))

    def test_pide_mas_de_los_que_hay(self):
        mensajes = []
        s = self.salida("pocos")
        rc = S.muestrear([S.Fuente(self.v1, step=40, desde="video|0.0|None")], s, por_video=50, formato="png", out=mensajes.append)
        self.assertEqual(rc, 0)
        self.assertEqual(self.manifiesto(s)["videos"][0]["frames"], [0, 40, 80])  # los 3 candidatos de la rejilla
        self.assertTrue(any("se pidieron 50" in m for m in mensajes))

    def test_un_video_inexistente_no_impide_los_demas(self):
        mensajes = []
        s = self.salida("faltante")
        rc = S.muestrear([S.Fuente("/no/existe.avi"), S.Fuente(self.v2, desde="video|0.0|None")], s, por_video=5, formato="png", out=mensajes.append)
        self.assertEqual(rc, 0)
        self.assertTrue(any("no se encuentra" in m for m in mensajes))
        self.assertEqual([v["nombre"] for v in self.manifiesto(s)["videos"]], ["dos.avi"])

    def test_si_nada_funciona_devuelve_2(self):
        s = self.salida("nada")
        self.assertEqual(S.muestrear([S.Fuente("/no/existe.avi")], s, out=silencio), 2)
        self.assertFalse(os.path.exists(os.path.join(s, "manifest.json")))  # no deja un manifiesto vacío
        self.assertEqual(S.muestrear([], s, out=silencio), 2)

    def test_un_archivo_que_no_es_video(self):
        falso = os.path.join(self.tmp.name, "falso.avi")
        with open(falso, "w") as fh:
            fh.write("esto no es un video")
        self.assertEqual(S.muestrear([S.Fuente(falso)], self.salida("falso"), out=silencio), 2)

    def test_argumentos_invalidos(self):
        f = [S.Fuente(self.v1, desde="video|0.0|None")]
        self.assertEqual(S.muestrear(f, self.salida("i1"), por_video=0, out=silencio), 2)
        self.assertEqual(S.muestrear(f, self.salida("i2"), formato="bmp", out=silencio), 2)

    def test_expande_carpetas_y_ignora_lo_que_no_es_video(self):
        carpeta = os.path.join(self.tmp.name, "mezcla")
        os.makedirs(carpeta)
        video_con_indices(carpeta, "a.avi", 30)
        for nombre in ("notas.txt", "foto.jpg"):
            with open(os.path.join(carpeta, nombre), "w") as fh:
                fh.write("x")
        fuentes = S.fuentes_desde_args([carpeta], [], paso=1, desde_s=0.0, hasta_s=None)
        self.assertEqual([os.path.basename(f.video) for f in fuentes], ["a.avi"])

    def test_entradas_ilegibles_se_avisan_y_se_omiten(self):
        mensajes = []
        self.assertEqual(S.fuentes_desde_args([], ["/no/hay/entradas"], 1, 0.0, None, out=mensajes.append), [])
        self.assertTrue(any("no se pudieron leer" in m for m in mensajes))

    def test_cli(self):
        s = self.salida("cli")
        rc = S.main([self.v2, "--salida", s, "--por-video", "6", "--formato", "png", "--semilla", "3"])
        self.assertEqual(rc, 0)
        self.assertEqual(len(self.manifiesto(s)["videos"][0]["frames"]), 6)


if __name__ == "__main__":
    unittest.main()