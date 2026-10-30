"""Pruebas de tools/eval_tracking.py.

Ejecutar desde la raíz del proyecto:
    python -m pytest tests/test_eval_tracking.py -v
"""

import csv
import importlib
import json
import math
import os
import tempfile
import unittest

import cv2
import numpy as np

from learning.dataset import yolo_label_line
from persistence.run_inputs import save_run_inputs
from pipeline import TrackingJob
from tools import eval_tracking as E
from tools import sample_frames as S
from tracking.background_tracker import DEFAULT_PARAMS, BackgroundTracker, compute_background
from video.frame_selection import FrameRange, compute_timing, read_background_frames
from video.reader import VideoReader


def _imp(name):
    try:
        return importlib.import_module(f"tests.{name}")
    except ImportError:
        return importlib.import_module(name)


RI = _imp("test_run_inputs")
BG = _imp("test_background_tracker")


def silencio(_):
    pass


def leer(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def etiquetar(valdir, vid, ancho, alto, cajas):
    """Escribe etiquetas YOLO. ``cajas``: {frame: (x, y, w, h) | None}; None = etiqueta vacía (sin ratón)."""
    os.makedirs(os.path.join(valdir, "labels"), exist_ok=True)
    for f, box in cajas.items():
        with open(os.path.join(valdir, "labels", E.image_stem(vid, f) + ".txt"), "w") as fh:
            if box is not None:
                fh.write(yolo_label_line(box, ancho, alto) + "\n")


def caja_circulo(cx, cy, r):
    return (int(cx) - r, int(cy) - r, 2 * r + 1, 2 * r + 1)


class EtiquetasTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def escribir(self, nombre, texto, bom=False):
        p = os.path.join(self.d, nombre)
        with open(p, "w", encoding="utf-8-sig" if bom else "utf-8") as fh:
            fh.write(texto)
        return p

    def test_caja_normal_con_numeros_calculados_a_mano(self):
        # xc=0.5*400=200, yc=0.5*300=150, w=0.25*400=100, h=0.5*300=150 -> x=200-50, y=150-75
        estado, box = E.parse_label_file(self.escribir("a.txt", "0 0.5 0.5 0.25 0.5\n"), 400, 300)
        self.assertEqual(estado, "caja")
        for got, want in zip(box, (150.0, 75.0, 100.0, 150.0)):
            self.assertAlmostEqual(got, want, places=6)

    def test_vacio_con_o_sin_espacios(self):
        for texto in ("", "\n", "  \n \n"):
            self.assertEqual(E.parse_label_file(self.escribir("v.txt", texto), 400, 300), ("vacio", None))

    def test_varias_cajas_e_invalidas(self):
        self.assertEqual(E.parse_label_file(self.escribir("m.txt", "0 .5 .5 .1 .1\n0 .2 .2 .1 .1\n"), 400, 300), ("varias", None))
        for basura in ("hola", "0 0.5 0.5", "0 a b c d"):
            self.assertEqual(E.parse_label_file(self.escribir("i.txt", basura), 400, 300), ("invalida", None), msg=basura)

    def test_tolera_bom_de_windows(self):
        estado, _ = E.parse_label_file(self.escribir("b.txt", "0 0.5 0.5 0.2 0.2\n", bom=True), 100, 100)
        self.assertEqual(estado, "caja")

    def manifiesto(self, frames):
        return {"videos": [{"id": "rata", "ancho": 400, "alto": 300, "frames": frames}]}

    def test_collect_truths_clasifica_cada_caso(self):
        val = os.path.join(self.d, "val")
        os.makedirs(os.path.join(val, "labels"))
        for nombre, texto in (("rata_f000001", "0 0.5 0.5 0.25 0.5\n"), ("rata_f000002", ""), ("rata_f000004", "0 .5 .5 .1 .1\n0 .2 .2 .1 .1\n"),
                              ("rata_f000005", "basura")):
            with open(os.path.join(val, "labels", nombre + ".txt"), "w") as fh:
                fh.write(texto)
        truths, info = E.collect_truths(val, self.manifiesto([1, 2, 3, 4, 5]))
        self.assertEqual(set(truths), {("rata", 1), ("rata", 2)})
        self.assertIsNone(truths[("rata", 2)])  # etiqueta vacía = sin ratón
        self.assertEqual((info["con_caja"], info["vacias"]), (1, 1))
        self.assertEqual(info["sin_etiqueta"], ["rata_f000003"])
        self.assertEqual(info["con_varias_cajas"], ["rata_f000004"])
        self.assertEqual(info["invalidas"], ["rata_f000005"])

    def test_faltantes_sin_raton_convierte_los_que_no_existen(self):
        val = os.path.join(self.d, "val2")
        os.makedirs(os.path.join(val, "labels"))
        truths, info = E.collect_truths(val, self.manifiesto([1, 2]), faltantes_sin_raton=True)
        self.assertEqual(truths, {("rata", 1): None, ("rata", 2): None})
        self.assertEqual(info["sin_etiqueta"], [])


class MetricasTests(unittest.TestCase):
    """Todos los números esperados están calculados a mano."""

    def datos(self):
        t = {
            "k1": (0, 0, 10, 10), "k2": (0, 0, 10, 10), "k3": (0, 0, 10, 10),
            "k4": None, "k5": None, "k6": (100, 100, 20, 20),
        }
        p = {
            "k1": E.Pred(True, (0, 0, 10, 10), (5, 5)),  # IoU 1, error 0
            "k2": E.Pred(True, (5, 5, 10, 10), (12, 12)),  # IoU 25/175, fuera de la caja, error dist((12,12),(5,5))
            "k3": E.Pred(False),  # perdido
            "k4": E.Pred(True, (50, 50, 10, 10), (55, 55)),  # falso positivo
            "k5": E.Pred(False),  # sin ratón y no dijo nada: bien
            "k6": E.Pred(True, (102, 100, 20, 20), (111, 110)),  # IoU 360/440, error 1
        }
        return t, p

    def test_estados_por_frame(self):
        t, p = self.datos()
        rows, _ = E.score_frames(t, p)
        estados = {r["clave"]: r["estado"] for r in rows}
        self.assertEqual(estados, {"k1": E.ST_OK, "k2": E.ST_WRONG, "k3": E.ST_MISS, "k4": E.ST_FP, "k5": E.ST_TN, "k6": E.ST_OK})

    def test_iou_y_errores_por_frame(self):
        t, p = self.datos()
        rows, _ = E.score_frames(t, p)
        r = {x["clave"]: x for x in rows}
        self.assertAlmostEqual(r["k1"]["iou"], 1.0)
        self.assertAlmostEqual(r["k2"]["iou"], 25 / 175)  # intersección 5x5, unión 100+100-25
        self.assertAlmostEqual(r["k6"]["iou"], 360 / 440)  # intersección 18x20, unión 400+400-360
        self.assertAlmostEqual(r["k2"]["error_centro_px"], math.sqrt(98))  # (12,12) a (5,5)
        self.assertAlmostEqual(r["k6"]["error_centro_px"], 1.0)  # (111,110) a (110,110)
        self.assertAlmostEqual(r["k6"]["error_centro_rel"], 1.0 / math.hypot(20, 20))
        self.assertEqual([r[k]["posicion_dentro"] for k in ("k1", "k2", "k6")], [True, False, True])

    def test_resumen(self):
        t, p = self.datos()
        _, s = E.score_frames(t, p)
        self.assertEqual((s["evaluados"], s["con_raton"], s["sin_raton"]), (6, 4, 2))
        self.assertEqual((s["aciertos"], s["lugar_equivocado"], s["perdidos"], s["falsos_positivos"]), (2, 1, 1, 1))
        self.assertAlmostEqual(s["acierto_pct"], 50.0)
        self.assertAlmostEqual(s["posicion_dentro_pct"], 50.0)  # k1 y k6 de 4
        self.assertAlmostEqual(s["perdidos_pct"], 25.0)
        self.assertAlmostEqual(s["falsos_positivos_pct"], 50.0)  # 1 de 2 imágenes sin ratón
        self.assertAlmostEqual(s["iou_mediano"], 360 / 440)  # mediana de [1, 25/175, 360/440]
        errs = sorted([0.0, math.sqrt(98), 1.0])
        self.assertAlmostEqual(s["error_centro_mediana_px"], 1.0)
        self.assertAlmostEqual(s["error_centro_p90_px"], errs[1] + 0.8 * (errs[2] - errs[1]))  # interpolación lineal, índice 1.8
        self.assertAlmostEqual(s["error_centro_max_px"], math.sqrt(98))

    def test_conversion_a_cm(self):
        t, p = self.datos()
        _, s = E.score_frames(t, p, cm_per_px=0.1)
        self.assertAlmostEqual(s["error_centro_mediana_cm"], 0.1)
        self.assertAlmostEqual(s["error_centro_max_cm"], math.sqrt(98) * 0.1)
        self.assertNotIn("error_centro_mediana_cm", E.score_frames(t, p)[1])  # sin escala no se inventa

    def test_umbral_de_iou(self):
        t, p = self.datos()
        self.assertEqual(E.score_frames(t, p, iou_thr=0.1)[1]["aciertos"], 3)  # k2 (0.143) ya cuenta
        self.assertEqual(E.score_frames(t, p, iou_thr=0.9)[1]["aciertos"], 1)  # solo k1

    def test_solo_se_evaluan_los_frames_con_prediccion_y_etiqueta(self):
        t, p = self.datos()
        del p["k6"]
        t["k7"] = (0, 0, 5, 5)  # etiquetado, pero el pipeline no lo procesó
        _, s = E.score_frames(t, p)
        self.assertEqual(s["evaluados"], 5)

    def test_sin_caja_de_prediccion_cuenta_la_posicion(self):
        t = {"a": (0, 0, 10, 10), "b": (0, 0, 10, 10)}
        p = {"a": E.Pred(True, None, (5, 5)), "b": E.Pred(True, None, (50, 50))}
        rows, _ = E.score_frames(t, p)
        self.assertEqual([r["estado"] for r in rows], [E.ST_OK, E.ST_WRONG])

    def test_sin_datos(self):
        _, s = E.score_frames({}, {})
        self.assertEqual(s["evaluados"], 0)
        self.assertIsNone(s["acierto_pct"])
        self.assertIsNone(s["error_centro_mediana_px"])

    def test_merge_equivale_a_evaluar_todo_junto(self):
        t, p = self.datos()
        a_t, a_p = {k: t[k] for k in ("k1", "k2", "k3")}, {k: p[k] for k in ("k1", "k2", "k3")}
        b_t, b_p = {k: t[k] for k in ("k4", "k5", "k6")}, {k: p[k] for k in ("k4", "k5", "k6")}
        ra, _ = E.score_frames(a_t, a_p)
        rb, _ = E.score_frames(b_t, b_p)
        for rows, pr in ((ra, a_p), (rb, b_p)):
            for r in rows:
                r["_pos"] = pr[r["clave"]].pos
        junto = E.merge_scored([ra, rb], 0.5)
        _, esperado = E.score_frames(t, p)
        for k in ("evaluados", "aciertos", "lugar_equivocado", "perdidos", "falsos_positivos", "posicion_dentro"):
            self.assertEqual(junto[k], esperado[k], msg=k)
        self.assertAlmostEqual(junto["error_centro_mediana_px"], esperado["error_centro_mediana_px"])


class SalidasTests(unittest.TestCase):
    def test_tabla(self):
        t, p = MetricasTests().datos()
        _, s = E.score_frames(t, p)
        txt = E.format_table([("fondo · v.avi", s), ("TOTAL fondo", s)])
        self.assertIn("fondo · v.avi", txt)
        self.assertIn("50 %", txt)
        self.assertIn("1/2", txt)  # falsos positivos / imágenes sin ratón
        self.assertEqual(len(txt.splitlines()), 3)

    def test_csv(self):
        t, p = MetricasTests().datos()
        rows, _ = E.score_frames(t, p)
        for r in rows:
            r.update(entradas="e", modo="fondo", video="v.avi", frame=0)
        with tempfile.TemporaryDirectory() as d:
            ruta = os.path.join(d, "x.csv")
            E.write_csv(ruta, rows)
            with open(ruta, newline="", encoding="utf-8-sig") as fh:
                filas = list(csv.DictReader(fh))
        self.assertEqual(len(filas), 6)
        self.assertEqual(filas[0]["estado"], E.ST_OK)
        self.assertEqual(filas[0]["posicion_dentro"], "si")
        self.assertEqual(filas[2]["iou"], "")  # el perdido no tiene IoU: celda vacía, no un número inventado
        self.assertEqual(filas[1]["caja_etiqueta"], "0.0 0.0 10.0 10.0")

    def test_match_video(self):
        man = {"videos": [{"id": "a", "nombre": "a.avi", "sha256_extremos": "AAA"}, {"id": "b", "nombre": "b.avi", "sha256_extremos": "BBB"}]}
        self.assertEqual(E.match_video(man, {"video": {"path": "/x/otro_nombre.avi", "sha256_extremos": "BBB"}})["id"], "b")  # huella manda
        self.assertEqual(E.match_video(man, {"video": {"path": "/x/a.avi"}})["id"], "a")  # sin huella: por nombre
        self.assertIsNone(E.match_video(man, {"video": {"path": "/x/c.avi", "sha256_extremos": "CCC"}}))

    def test_hoja_de_fallos(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "images"))
            man = {"videos": [{"id": "v", "formato": "png"}]}
            for f in (1, 2):
                cv2.imwrite(os.path.join(d, "images", E.image_stem("v", f) + ".png"), np.full((60, 80, 3), 100, np.uint8))
            ok = {"clave": ("v", 1), "estado": E.ST_OK, "caja_etiqueta": (5, 5, 10, 10), "_pos": (10, 10)}
            malo = {"clave": ("v", 2), "estado": E.ST_WRONG, "caja_etiqueta": (5, 5, 10, 10), "_pos": (60, 40), "error_centro_px": 50.0}
            self.assertFalse(E.write_failure_sheet(os.path.join(d, "f.png"), [ok], d, man))  # sin fallos no se escribe nada
            self.assertFalse(os.path.exists(os.path.join(d, "f.png")))
            self.assertTrue(E.write_failure_sheet(os.path.join(d, "f.png"), [ok, malo], d, man))
            img = cv2.imread(os.path.join(d, "f.png"))
            self.assertIsNotNone(img)
            self.assertEqual(img.shape[1], 360)  # una celda de 360 px de ancho
            self.assertGreater(int((img[:, :, 2] > 200).sum()), 0)  # la cruz roja está dibujada


class EscenaFacilTests(unittest.TestCase):
    """Video sintético fácil (círculo brillante de radio 13) con verdad conocida; modo fondo."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        RI.Escenas.preparar(cls.tmp.name)
        cls.frames, cls.truth = BG.make_frames(90, lost={30, 31, 32})
        with VideoReader(RI.Escenas.p_bg) as r:
            job = RI.Escenas.job_fondo(r)  # rango 0..85, frame_step 2
            cls.carpeta = save_run_inputs(os.path.join(cls.tmp.name, "exp"), job, video_path=RI.Escenas.p_bg, options=RI.Escenas.OPT_FONDO,
                                          sample_idx=0, app_version="1.0", log=None, now=RI.AHORA)
        cls.n = 0

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def preparar(self, por_video=100, etiquetas=True):
        """Conjunto con TODOS los frames procesables (rejilla de paso 2) y etiquetas hechas desde la verdad."""
        type(self).n += 1
        val = os.path.join(self.tmp.name, f"val{self.n}")
        self.assertEqual(S.muestrear(S.fuentes_desde_args([], [self.carpeta], 1, 0.0, None), val, por_video=por_video, out=silencio), 0)
        man = json.loads(leer(os.path.join(val, "manifest.json")))
        v = man["videos"][0]
        if etiquetas:
            etiquetar(val, v["id"], v["ancho"], v["alto"],
                      {f: (None if self.truth[f] is None else caja_circulo(*self.truth[f], 13)) for f in v["frames"]})
        return val, v

    def correr(self, val, **kw):
        mensajes = []
        rc = E.run(val, [self.carpeta], out=mensajes.append, **kw)
        return rc, mensajes

    def resultados(self, val):
        carpeta = sorted(d for d in os.listdir(val) if d.startswith("evaluacion_"))[-1]
        return os.path.join(val, carpeta), json.loads(leer(os.path.join(val, carpeta, "resultados.json")))

    def test_sin_filtros_todo_acierta_y_el_error_es_de_un_pixel(self):
        val, v = self.preparar()
        rc, _ = self.correr(val, sin_filtros=True)
        self.assertEqual(rc, 0)
        _, res = self.resultados(val)
        s = res["corridas"][0]["resumen"]
        self.assertEqual(s["evaluados"], len(v["frames"]))
        self.assertEqual((s["acierto_pct"], s["posicion_dentro_pct"]), (100.0, 100.0))
        self.assertEqual((s["lugar_equivocado"], s["perdidos"], s["falsos_positivos"]), (0, 0, 0))
        self.assertGreaterEqual(s["sin_raton"], 1)  # los frames 30 y 32 no tienen ratón y se etiquetaron vacíos
        self.assertLess(s["error_centro_max_px"], 2.0)

    def test_el_suavizado_se_mide_como_retraso(self):
        """Con α = 0.35 la posición del CSV va detrás del ratón ~ (1-α)/α pasos; sin filtros no."""
        val, _ = self.preparar()
        self.correr(val, sin_filtros=True)
        _, sin = self.resultados(val)
        self.correr(val)
        _, con = self.resultados(val)
        e_sin = sin["corridas"][0]["resumen"]["error_centro_mediana_px"]
        e_con = con["corridas"][0]["resumen"]["error_centro_mediana_px"]
        paso_px = 2.0 * 2  # el círculo avanza 2 px por frame y se procesa uno de cada 2
        retraso_esperado = paso_px * (1 - 0.35) / 0.35  # ≈ 7.4 px solo en x
        self.assertGreater(e_con, 0.8 * retraso_esperado)
        self.assertLess(e_con, 2.0 * retraso_esperado)
        self.assertGreater(e_con, 4 * e_sin)
        self.assertEqual(con["corridas"][0]["resumen"]["acierto_pct"], 100.0)  # la caja del tracker (sin filtrar) sigue acertando

    def test_falso_positivo_perdido_y_lugar_equivocado(self):
        val, v = self.preparar()
        base = self.correr(val, sin_filtros=True)
        self.assertEqual(base[0], 0)
        _, r0 = self.resultados(val)
        s0 = r0["corridas"][0]["resumen"]
        vid, a, b = v["id"], v["ancho"], v["alto"]
        etiquetar(val, vid, a, b, {
            30: (150, 100, 27, 27),  # no hay ratón (hueco del video), pero decimos que sí: el seguimiento no lo ve -> perdido
            40: None,  # sí hay ratón, pero decimos que no -> falso positivo
            50: (self.truth[50][0] + 120, self.truth[50][1], 27, 27),  # caja lejos del ratón real -> lugar equivocado
        })
        self.correr(val, sin_filtros=True)
        _, r1 = self.resultados(val)
        s1 = r1["corridas"][0]["resumen"]
        self.assertEqual((s1["perdidos"], s1["falsos_positivos"], s1["lugar_equivocado"]), (1, 1, 1))
        self.assertEqual(s1["aciertos"], s0["aciertos"] - 2)
        self.assertEqual(s1["con_raton"], s0["con_raton"])  # +1 (frame 30) y -1 (frame 40)

    def test_escribe_resultados_csv_y_hoja_de_fallos(self):
        val, v = self.preparar()
        etiquetar(val, v["id"], v["ancho"], v["alto"], {40: None})
        self.correr(val, sin_filtros=True)
        carpeta, res = self.resultados(val)
        with open(os.path.join(carpeta, "por_frame.csv"), newline="", encoding="utf-8-sig") as fh:
            filas = list(csv.DictReader(fh))
        self.assertEqual(len(filas), res["corridas"][0]["resumen"]["evaluados"])
        self.assertEqual([f for f in filas if f["frame"] == "40"][0]["estado"], E.ST_FP)
        self.assertEqual(res["corridas"][0]["modo"], "fondo")
        self.assertIn("fondo", res["por_modo"])
        self.assertIsNotNone(cv2.imread(os.path.join(carpeta, "fallos.png")))

    def test_sin_fallos_no_hay_hoja(self):
        val, _ = self.preparar()
        self.correr(val, sin_filtros=True)
        carpeta, _ = self.resultados(val)
        self.assertFalse(os.path.exists(os.path.join(carpeta, "fallos.png")))

    def test_etiquetas_que_faltan_se_omiten_y_se_cuentan(self):
        val, v = self.preparar(etiquetas=False)
        etiquetar(val, v["id"], v["ancho"], v["alto"], {f: caja_circulo(*self.truth[f], 13) for f in v["frames"][:10] if self.truth[f]})
        rc, mensajes = self.correr(val, sin_filtros=True)
        self.assertEqual(rc, 0)
        self.assertTrue(any("sin etiqueta (se omiten)" in m for m in mensajes))
        _, res = self.resultados(val)
        self.assertLessEqual(res["corridas"][0]["resumen"]["evaluados"], 10)

    def test_etiquetas_con_varias_cajas_se_avisan_y_no_se_evaluan(self):
        val, v = self.preparar()
        with open(os.path.join(val, "labels", E.image_stem(v["id"], 20) + ".txt"), "w") as fh:
            fh.write("0 .5 .5 .1 .1\n0 .2 .2 .1 .1\n")
        rc, mensajes = self.correr(val, sin_filtros=True)
        self.assertEqual(rc, 0)
        self.assertTrue(any("más de una caja" in m for m in mensajes))
        _, res = self.resultados(val)
        self.assertEqual(res["corridas"][0]["resumen"]["evaluados"], len(v["frames"]) - 1)

    def test_frames_que_el_pipeline_no_procesa_se_cuentan_aparte(self):
        """Muestreo con paso 1 sobre una corrida de frame_step 2: los impares no se procesan."""
        type(self).n += 1
        val = os.path.join(self.tmp.name, f"val{self.n}")
        S.muestrear([S.Fuente(RI.Escenas.p_bg, step=1, desde="video|0.0|None")], val, por_video=60, out=silencio)
        v = json.loads(leer(os.path.join(val, "manifest.json")))["videos"][0]
        etiquetar(val, v["id"], v["ancho"], v["alto"], {f: (None if self.truth[f] is None else caja_circulo(*self.truth[f], 13)) for f in v["frames"]})
        rc, mensajes = self.correr(val, sin_filtros=True)
        self.assertEqual(rc, 0)
        impares = [f for f in v["frames"] if f % 2 == 1 or f > 85]
        self.assertGreater(len(impares), 0)
        self.assertTrue(any("no se procesan" in m for m in mensajes))
        _, res = self.resultados(val)
        self.assertEqual(res["corridas"][0]["no_procesados"], len(impares))
        self.assertEqual(res["corridas"][0]["resumen"]["evaluados"], len(v["frames"]) - len(impares))

    def test_varias_corridas_se_totalizan_por_modo(self):
        val, v = self.preparar()
        mensajes = []
        rc = E.run(val, [self.carpeta, self.carpeta], sin_filtros=True, out=mensajes.append)
        self.assertEqual(rc, 0)
        _, res = self.resultados(val)
        self.assertEqual(len(res["corridas"]), 2)
        self.assertEqual(res["por_modo"]["fondo"]["evaluados"], 2 * len(v["frames"]))
        self.assertTrue(any(m.startswith("TOTAL fondo") or "TOTAL fondo" in m for m in mensajes))

    def test_una_corrida_de_otro_video_se_omite(self):
        val, _ = self.preparar()
        with VideoReader(RI.Escenas.p_col) as r:
            otra = save_run_inputs(os.path.join(self.tmp.name, "exp_col"), RI.Escenas.job_color(r), video_path=RI.Escenas.p_col,
                                   options=RI.Escenas.OPT_COL, sample_idx=0, app_version="1.0", log=None, now=RI.AHORA)
        mensajes = []
        self.assertEqual(E.run(val, [otra], out=mensajes.append), 2)
        self.assertTrue(any("no está en el conjunto" in m for m in mensajes))

    def test_errores_de_entrada(self):
        val, _ = self.preparar()
        self.assertEqual(E.run(val, [], out=silencio), 2)  # sin corridas
        self.assertEqual(E.run(os.path.join(self.tmp.name, "no_existe"), [self.carpeta], out=silencio), 2)  # sin manifiesto
        vacio, _ = self.preparar(etiquetas=False)
        self.assertEqual(E.run(vacio, [self.carpeta], out=silencio), 2)  # sin etiquetas
        self.assertEqual(E.run(val, ["/no/hay/entradas"], out=silencio), 2)  # entradas ilegibles

    def test_faltantes_sin_raton(self):
        val, _ = self.preparar(etiquetas=False)
        self.assertEqual(E.run(val, [self.carpeta], faltantes_sin_raton=True, sin_filtros=True, out=silencio), 0)
        _, res = self.resultados(val)
        s = res["corridas"][0]["resumen"]
        self.assertEqual(s["con_raton"], 0)  # todo cuenta como «sin ratón»: los frames con ratón visible son falsos positivos
        self.assertGreater(s["falsos_positivos"], 0)

    def test_cli(self):
        val, _ = self.preparar()
        self.assertEqual(E.main([val, "--entradas", self.carpeta, "--sin-filtros", "--hoja-fallos", "0"]), 0)


class FantasmaDelFondoTests(unittest.TestCase):
    """El fallo silencioso que se encontró: el ratón quieto al elegir al sujeto se absorbe en el fondo y la caja
    se queda pegada a su sitio original. El informe de calidad no lo ve; este evaluador sí."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(1)
        base = (np.linspace(60, 110, 320)[None, :] + np.zeros((240, 1))).astype(np.float32)
        cls.pos = lambda self, i: (60.0, 120.0) if i < 60 else (60.0 + 200.0 * min(1.0, (i - 60) / 100.0), 120.0)  # noqa: E731
        frames = []
        for i in range(240):
            f = cv2.cvtColor(np.clip(base + rng.normal(0, 2.0, (240, 320)), 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            x, y = cls.pos(cls, i)
            cv2.circle(f, (int(round(x)), int(round(y))), 11, (230, 230, 230), -1)
            frames.append(f)
        cls.video = RI.escribir_video(cls.tmp.name, "fantasma.avi", frames)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def entradas(self, nombre, ref):
        """Corrida de modo fondo con el sujeto elegido en el frame ``ref`` (el fondo se calcula alrededor de él)."""
        mask, roi = np.full((240, 320), 255, np.uint8), (0, 0, 320, 240)
        with VideoReader(self.video) as r:
            bg = compute_background(read_background_frames(r, ref), roi, mask, False)
            tr = BackgroundTracker(mask, bg, dict(DEFAULT_PARAMS), polarity="white_on_black", apply_bw=False, use_clahe=False)
            x0, y0 = self.pos(ref)
            job = TrackingJob(r, FrameRange(ref, 239), compute_timing(r.fps), tr, roi, start=((x0, y0), (int(x0) - 15, int(y0) - 15, 30, 30)))
            # las opciones deben coincidir con el trabajo (se procesa cada frame): si no, build_job lo rechaza
            opciones = {k: v for k, v in RI.Escenas.OPT_FONDO.items() if k != "tracking_fps"}
            opciones["custom_tracking_fps"] = False
            return save_run_inputs(os.path.join(self.tmp.name, nombre), job, video_path=self.video, options=opciones,
                                   sample_idx=ref, app_version="1.0", log=None, now=RI.AHORA)

    def evaluar(self, nombre, ref):
        carpeta = self.entradas(nombre, ref)
        val = os.path.join(self.tmp.name, "val_" + nombre)
        S.muestrear(S.fuentes_desde_args([], [carpeta], 1, 0.0, None), val, por_video=80, out=silencio)
        v = json.loads(leer(os.path.join(val, "manifest.json")))["videos"][0]
        # etiquetamos solo la parte en que el ratón ya cruza la arena (frames >= 130), donde el fantasma se nota
        etiquetar(val, v["id"], v["ancho"], v["alto"], {f: caja_circulo(*self.pos(f), 11) for f in v["frames"] if f >= 130})
        mensajes = []
        self.assertEqual(E.run(val, [carpeta], sin_filtros=True, out=mensajes.append), 0)
        res = json.loads(leer(os.path.join(val, sorted(d for d in os.listdir(val) if d.startswith("evaluacion_"))[-1], "resultados.json")))
        return res["corridas"][0]["resumen"]

    def test_elegir_con_el_raton_quieto_deja_la_caja_pegada(self):
        s = self.evaluar("quieto", ref=30)
        self.assertGreater(s["con_raton"], 10)
        self.assertLess(s["acierto_pct"], 30.0)
        self.assertGreater(s["error_centro_mediana_px"], 80.0)  # el ratón recorre ~200 px; la caja sigue en x ≈ 60

    def test_elegir_con_el_raton_en_movimiento_funciona(self):
        s = self.evaluar("movimiento", ref=110)
        self.assertEqual(s["acierto_pct"], 100.0)
        self.assertLess(s["error_centro_max_px"], 3.0)


if __name__ == "__main__":
    unittest.main()