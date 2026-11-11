"""Pruebas de tools/scan_video.py (escanea un video completo con el detector y marca los casos difíciles).

Ejecutar desde la raíz del proyecto:
    python -m pytest tests/test_scan_video.py -v

El análisis se prueba con muestras hechas a mano; el recorrido del video, con un video sintético REAL y el detector de juguete que lee
el tensor de entrada (el de tests/test_eval_detector.py).
"""

import csv
import importlib
import json
import os
import shutil
import unittest
from unittest import mock

import cv2
import numpy as np

from tools import eval_detector as ED
from tools import scan_video as SV


def _imp(nombre):
    try:
        return importlib.import_module(f"tests.{nombre}")
    except ImportError:
        return importlib.import_module(nombre)


TD = _imp("test_eval_detector")
TM, BG = TD.TM, TD.BG
CROP = (0, 0, 200, 200)


def leer_json(ruta):
    with open(ruta, encoding="utf-8") as fh:
        return json.load(fh)


def sample(idx, dets, crop=CROP, ms=10.0, t=None):
    """Muestra a mano: ``dets`` es una lista de (confianza, (x, y, ancho, alto))."""
    return SV.Sample(idx, float(idx if t is None else t), ms, [SV.Det(s, b) for s, b in dets], crop)


def caja(cx, cy, w=30, h=30):
    return (cx - w / 2, cy - h / 2, w, h)


def tira(n, score=0.95, w=30, paso=4, x0=40):
    """n muestras con una caja centrada que avanza ``paso`` px por imagen, lejos de los bordes."""
    return [sample(i, [(score, caja(x0 + i * paso, 100, w, w))]) for i in range(n)]


# ======================================================================
class MarcasTests(unittest.TestCase):
    def marcas(self, muestras, umbral=0.5, confianza=0.8):
        return SV.analyze(muestras, umbral, confianza)

    def test_sin_deteccion(self):
        m = self.marcas([sample(0, []), sample(1, [(0.3, caja(100, 100))]), sample(2, [(0.5, caja(100, 100))]), sample(3, [(0.51, caja(100, 100))])])
        self.assertEqual(m[0], ("SIN_DETECCION",))
        self.assertEqual(m[1], ("SIN_DETECCION",))  # hay un candidato débil, pero no supera el umbral
        self.assertEqual(m[2], ("SIN_DETECCION",))  # la igualdad no alcanza: el umbral es estricto
        self.assertEqual(m[3], ("CONFIANZA_BAJA",))

    def test_confianza_baja_es_estricta(self):
        m = self.marcas([sample(0, [(0.79, caja(100, 100))]), sample(1, [(0.80, caja(100, 100))]), sample(2, [(0.95, caja(100, 100))])])
        self.assertEqual([x for x in m], [("CONFIANZA_BAJA",), (), ()])

    def test_varias(self):
        lejos = sample(0, [(0.95, caja(50, 50)), (0.6, caja(150, 150))])  # otro candidato lejos y con confianza suficiente
        mucho = sample(1, [(0.95, caja(100, 100)), (0.9, caja(102, 100))])  # casi la misma caja (IoU alto): es el mismo ratón
        debil = sample(2, [(0.95, caja(50, 50)), (0.4, caja(150, 150))])  # el otro candidato no pasa el umbral
        borde = sample(3, [(0.95, caja(50, 50)), (0.5, caja(150, 150))])  # igual al umbral: no cuenta
        m = self.marcas([lejos, mucho, debil, borde])
        self.assertEqual(m[0], ("VARIAS",))
        self.assertEqual((m[1], m[2], m[3]), ((), (), ()))

    def test_varias_el_limite_del_iou_de_0_3(self):
        """Cajas de 40x40; la principal en x 80..120. Se desplaza la otra horizontalmente y se calcula el IoU a mano."""
        a = (80, 80, 40, 40)

        def con_desplazamiento(dx):
            return self.marcas([sample(0, [(0.95, a), (0.9, (80 + dx, 80, 40, 40))])])[0]
        # dx = 9: solape 31x40 = 1240; unión 3200 - 1240 = 1960; IoU 0.633 -> el mismo ratón
        self.assertEqual(con_desplazamiento(9), ())
        # dx = 20: solape 20x40 = 800; unión 2400; IoU 0.333 -> todavía el mismo
        self.assertEqual(con_desplazamiento(20), ())
        # dx = 22: solape 18x40 = 720; unión 2480; IoU 0.290 -> otro
        self.assertEqual(con_desplazamiento(22), ("VARIAS",))
        # sin solape: otro
        self.assertEqual(con_desplazamiento(60), ("VARIAS",))

    def test_caja_rara_por_area(self):
        muestras = tira(10) + [sample(10, [(0.95, caja(120, 100, 90, 90))])]  # 90x90 = 8100 contra áreas de 900
        m = self.marcas(muestras)
        self.assertEqual(m[10], ("CAJA_RARA", "SALTO") if "SALTO" in m[10] else ("CAJA_RARA",))
        self.assertTrue(all("CAJA_RARA" not in x for x in m[:10]))

    def test_caja_rara_tambien_por_chica(self):
        m = self.marcas(tira(10) + [sample(10, [(0.95, caja(80, 100, 4, 4))])])
        self.assertIn("CAJA_RARA", m[10])

    def test_con_pocos_datos_no_hay_caja_rara_ni_salto(self):
        muestras = tira(5) + [sample(5, [(0.95, caja(100, 100, 90, 90))]), sample(6, [(0.95, caja(180, 100, 90, 90))])]
        m = self.marcas(muestras)
        self.assertTrue(all("CAJA_RARA" not in x and "SALTO" not in x for x in m))  # menos de 8 datos: no hay base para decir qué es habitual

    def test_el_minimo_de_datos_para_decir_que_es_habitual_es_exactamente_ocho(self):
        """6 normales y 1 enorme (7 datos): sin base para decir qué es raro. Con una más (8 datos) ya sí: es justo el límite."""
        normales = tira(6)
        grande = lambda i: sample(i, [(0.95, caja(100, 100, 90, 90))])  # noqa: E731
        self.assertTrue(all("CAJA_RARA" not in x for x in self.marcas(normales + [grande(6)])))  # 7 datos
        m = self.marcas(normales + [sample(6, [(0.95, caja(100, 100, 30, 30))]), grande(7)])  # 8 datos
        self.assertIn("CAJA_RARA", m[7])

    def test_las_de_baja_confianza_no_definen_lo_habitual(self):
        """Las áreas de referencia salen solo de detecciones seguras: las dudosas no deben correr los límites."""
        seguras = tira(8)
        dudosas = [sample(8 + i, [(0.6, caja(100, 100, 100, 100))]) for i in range(6)]  # 6 cajas enormes pero dudosas
        m = self.marcas(seguras + dudosas)
        self.assertTrue(all("CAJA_RARA" in x for x in m[8:]))  # siguen siendo raras frente a las seguras

    def test_salto(self):
        # 12 imágenes con pasos de 4 px y luego un salto de 100 px, mayor que el ancho de la caja (30)
        muestras = tira(12) + [sample(12, [(0.95, caja(40 + 11 * 4 + 100, 100))])]
        m = self.marcas(muestras)
        self.assertIn("SALTO", m[12])
        self.assertTrue(all("SALTO" not in x for x in m[:12]))

    def test_un_ratón_que_se_mueve_rapido_de_forma_constante_no_salta(self):
        """Pasos de 40 px con cajas de 30 px: cada paso supera el ancho de la caja, pero es lo HABITUAL en este video, no un salto."""
        muestras = [sample(i, [(0.95, caja(20 + i * 40, 100))], crop=(0, 0, 1000, 200)) for i in range(12)]
        self.assertTrue(all("SALTO" not in x for x in self.marcas(muestras)))

    def test_un_paso_grande_pero_menor_que_la_caja_no_es_salto(self):
        muestras = tira(12) + [sample(12, [(0.95, caja(40 + 11 * 4 + 25, 100))])]  # 25 px de paso (< 30 de ancho) pero fuera de lo habitual (4)
        self.assertNotIn("SALTO", self.marcas(muestras)[12])

    def test_no_hay_salto_sin_detección_previa(self):
        muestras = tira(12) + [sample(12, []), sample(13, [(0.95, caja(250, 100))])]  # tras un hueco no se mide el paso
        self.assertNotIn("SALTO", self.marcas(muestras)[13])

    def test_borde(self):
        pegada = sample(0, [(0.95, (0, 50, 30, 30))])  # x = 0: toca el borde izquierdo
        a_2 = sample(1, [(0.95, (2, 50, 30, 30))])  # a 2 px: cuenta
        a_3 = sample(2, [(0.95, (3, 50, 30, 30))])  # a 3 px: no
        derecha = sample(3, [(0.95, (170, 50, 30, 30))])  # x + w = 200 = borde derecho
        arriba = sample(4, [(0.95, (50, 1, 30, 30))])
        abajo = sample(5, [(0.95, (50, 170, 30, 30))])
        m = self.marcas([pegada, a_2, a_3, derecha, arriba, abajo])
        self.assertEqual([("BORDE" in x) for x in m], [True, True, False, True, True, True])

    def test_el_borde_se_mide_contra_el_recorte_no_contra_el_frame(self):
        recorte = (300, 50, 500, 250)
        m = self.marcas([sample(0, [(0.95, (302, 80, 30, 30))], crop=recorte),  # a 2 px del borde izquierdo del recorte
                         sample(1, [(0.95, (350, 100, 30, 30))], crop=recorte)])  # en el centro del recorte: no toca nada
        self.assertIn("BORDE", m[0])
        self.assertNotIn("BORDE", m[1])  # (contra un recorte de 0 a 200 este mismo cuadro sí «tocaría» el borde derecho: error de referencia)

    def test_las_marcas_salen_de_mas_a_menos_grave(self):
        muestras = tira(10) + [sample(10, [(0.7, caja(120, 100, 90, 90)), (0.6, caja(30, 30))])]  # baja confianza + varias + rara
        m = self.marcas(muestras)[10]
        self.assertEqual(m[:3], ("VARIAS", "CONFIANZA_BAJA", "CAJA_RARA"))
        self.assertEqual(list(m), sorted(m, key=SV.SEVERIDAD.index))

    def test_una_imagen_sana_no_tiene_marcas(self):
        self.assertEqual(self.marcas(tira(12)), [()] * 12)

    def test_primary(self):
        self.assertEqual(SV.primary(("VARIAS", "SALTO")), "VARIAS")
        self.assertIsNone(SV.primary(()))


# ======================================================================
class TramosYSeleccionTests(unittest.TestCase):
    def test_tramos_sin_deteccion(self):
        ok = [(0.95, caja(100, 100))]
        muestras = [sample(i, ok if i not in (3, 4, 5, 8, 12, 13) else []) for i in range(16)]
        marcas = SV.analyze(muestras, 0.5, 0.8)
        g = SV.gaps(muestras, marcas)
        self.assertEqual(g, [{"desde_s": 3.0, "hasta_s": 5.0, "imagenes": 3}, {"desde_s": 12.0, "hasta_s": 13.0, "imagenes": 2}])  # la imagen suelta (8) no es un tramo
        self.assertEqual(len(SV.gaps(muestras, marcas, minimo=1)), 3)
        self.assertEqual(SV.gaps(muestras, marcas, minimo=4), [])

    def test_un_tramo_al_final(self):
        ok = [(0.95, caja(100, 100))]
        muestras = [sample(i, ok if i < 4 else []) for i in range(7)]
        self.assertEqual(SV.gaps(muestras, SV.analyze(muestras, 0.5, 0.8)), [{"desde_s": 4.0, "hasta_s": 6.0, "imagenes": 3}])

    def test_seleccion_por_turnos_entre_los_tipos(self):
        marcas = [("SIN_DETECCION",)] * 6 + [("VARIAS",)] * 2 + [("CONFIANZA_BAJA",)] * 3 + [()] * 3 + [("BORDE",)] * 2 + [("SALTO", "BORDE")]
        elegidos = SV.select_cases(marcas, 6)
        # por turnos: una de cada tipo y luego otra vuelta (SIN_DETECCION, VARIAS, CONFIANZA_BAJA, SALTO, y de nuevo las tres primeras...)
        tipos = [SV.primary([m for m in marcas[i] if m != "BORDE"]) for i in elegidos]
        self.assertEqual(tipos.count("SIN_DETECCION"), 2)
        self.assertEqual(tipos.count("VARIAS"), 2)
        self.assertEqual(tipos.count("CONFIANZA_BAJA"), 1)
        self.assertEqual(tipos.count("SALTO"), 1)
        self.assertEqual(len(elegidos), 6)
        self.assertEqual(elegidos, sorted(elegidos, key=lambda i: (SV.SEVERIDAD.index(SV.primary([m for m in marcas[i] if m != "BORDE"])), i)))
        self.assertEqual(len(SV.select_cases(marcas, 100)), 12)  # los 6 + 2 + 3 + 1 con problema; las de solo BORDE y las sanas no entran

    def test_un_tipo_muy_frecuente_no_acapara_la_hoja(self):
        marcas = [("SIN_DETECCION",)] * 50 + [("VARIAS",)]
        elegidos = SV.select_cases(marcas, 5)
        self.assertIn(50, elegidos)  # la única con VARIAS entra aunque haya 50 sin detección

    def test_dentro_de_cada_tipo_van_en_orden_de_tiempo(self):
        marcas = [(), ("SIN_DETECCION",), (), ("SIN_DETECCION",), ("SIN_DETECCION",)]
        self.assertEqual(SV.select_cases(marcas, 2), [1, 3])

    def test_sin_problemas_no_hay_seleccion(self):
        self.assertEqual(SV.select_cases([(), ("BORDE",), ()], 10), [])
        self.assertEqual(SV.select_cases([("SIN_DETECCION",)], 0), [])

    def test_resumen(self):
        ok = (0.9, caja(100, 100))
        muestras = [sample(i, [ok], ms=100.0 + i) for i in range(6)] + [sample(6, [], ms=500.0), sample(7, [(0.7, caja(100, 100))])]
        marcas = SV.analyze(muestras, 0.5, 0.8)
        r = SV.summarize(muestras, marcas, 0.5, 0.8, 1.0)
        self.assertEqual((r["imagenes"], r["con_deteccion"]), (8, 7))
        self.assertAlmostEqual(r["con_deteccion_pct"], 87.5)
        self.assertEqual((r["marcas"]["SIN_DETECCION"], r["marcas"]["CONFIANZA_BAJA"]), (1, 1))
        self.assertEqual(r["imagenes_con_problemas"], 2)
        self.assertAlmostEqual(r["imagenes_con_problemas_pct"], 25.0)
        self.assertAlmostEqual(r["confianza_mediana"], 0.9)
        self.assertEqual(r["ms_mediana"], statistics_median([102.0, 103.0, 104.0, 105.0, 500.0, 10.0]))  # sin las dos primeras
        self.assertEqual(r["tramos_sin_deteccion"], [])  # una sola imagen sin detección no es un tramo

    def test_resumen_sin_detecciones(self):
        r = SV.summarize([sample(0, []), sample(1, [])], SV.analyze([sample(0, []), sample(1, [])], 0.5, 0.8), 0.5, 0.8, 1.0)
        self.assertEqual((r["con_deteccion"], r["confianza_mediana"], r["confianza_p10"]), (0, None, None))
        self.assertEqual(len(r["tramos_sin_deteccion"]), 1)

    def test_mmss(self):
        self.assertEqual((SV.mmss(0), SV.mmss(59.4), SV.mmss(59.6), SV.mmss(125), SV.mmss(3725), SV.mmss(-3)), ("0:00", "0:59", "1:00", "2:05", "62:05", "0:00"))


def statistics_median(vals):
    import statistics
    return statistics.median(vals)


# ======================================================================
class SalidasTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.t = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def con_miniatura(self, idx, dets, crop=(100, 50, 300, 250), escala=0.5, color=100):
        w, h = int((crop[2] - crop[0]) * escala), int((crop[3] - crop[1]) * escala)
        ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), color, np.uint8))
        s = sample(idx, dets, crop)
        s.thumb, s.escala = buf.tobytes(), escala
        return s

    def test_csv(self):
        muestras = [sample(10, [(0.9, (20, 30, 40, 50)), (0.6, (100, 100, 10, 10))], ms=12.34), sample(20, [(0.3, (1, 1, 5, 5))]), sample(30, [])]
        marcas = SV.analyze(muestras, 0.5, 0.8)
        ruta = os.path.join(self.t, "e.csv")
        SV.write_csv(ruta, muestras, marcas, 0.5)
        with open(ruta, newline="", encoding="utf-8-sig") as fh:
            f = list(csv.DictReader(fh))
        self.assertEqual(list(f[0]), ["frame", "tiempo_s", "ms", "n_candidatos", "confianza", "confianza_2", "x", "y", "ancho", "alto", "cx", "cy", "area", "marcas"])
        self.assertEqual((f[0]["frame"], f[0]["n_candidatos"], f[0]["confianza"], f[0]["confianza_2"]), ("10", "2", "0.900", "0.600"))
        self.assertEqual((f[0]["x"], f[0]["y"], f[0]["ancho"], f[0]["alto"], f[0]["cx"], f[0]["cy"], f[0]["area"]), ("20.0", "30.0", "40.0", "50.0", "40.0", "55.0", "2000"))
        self.assertEqual(f[0]["ms"], "12.3")
        self.assertIn("VARIAS", f[0]["marcas"])
        self.assertEqual((f[1]["confianza"], f[1]["x"], f[1]["marcas"]), ("0.300", "", "SIN_DETECCION"))  # el candidato débil se informa, sin caja
        self.assertEqual((f[2]["n_candidatos"], f[2]["confianza"]), ("0", ""))

    def test_hoja_tamano_y_orden_de_filas(self):
        muestras = [self.con_miniatura(i, [(0.95, (150, 100, 60, 60))]) for i in range(7)]
        marcas = [("SIN_DETECCION",)] * 7
        hoja = SV.render_sheet(muestras, marcas, list(range(7)), 0.5, 0.8, columnas=3, celda=100)  # cada miniatura mide 100 x 100
        self.assertEqual(hoja.shape, (3 * 100, 3 * 100, 3))  # 7 celdas en 3 columnas = 3 filas

    def test_la_hoja_vacia_es_none(self):
        self.assertIsNone(SV.render_sheet([], [], [], 0.5, 0.8))

    def test_colores_de_las_cajas(self):
        """Verde: principal con confianza suficiente. Amarillo: principal dudosa. Rojo: los otros candidatos."""
        def pix(img, color):
            return int(np.all(img == np.array(color, np.uint8), axis=2).sum())
        buena = self.con_miniatura(0, [(0.95, (150, 100, 60, 60))])
        dudosa = self.con_miniatura(1, [(0.65, (150, 100, 60, 60))])
        varias = self.con_miniatura(2, [(0.95, (150, 100, 40, 40)), (0.7, (250, 180, 30, 30))])
        for k, (s, g, y, r) in enumerate(((buena, True, False, False), (dudosa, False, True, False), (varias, True, False, True))):
            hoja = SV.render_sheet([s], [()], [0], 0.5, 0.8, columnas=1, celda=200)
            self.assertEqual((pix(hoja, SV.COL_BUENA) > 50, pix(hoja, SV.COL_DUDOSA) > 50, pix(hoja, SV.COL_OTRA) > 50), (g, y, r), msg=f"caso {k}")

    def test_la_caja_se_dibuja_donde_esta_aunque_el_recorte_no_empiece_en_cero(self):
        # recorte desde (100, 50), miniatura a escala 0.5; caja en el frame (150, 100, 60, 60) -> en la miniatura (25, 25) a (55, 55)
        s = self.con_miniatura(0, [(0.95, (150, 100, 60, 60))])
        hoja = SV.render_sheet([s], [()], [0], 0.5, 0.8, columnas=1, celda=100)  # la miniatura mide 100 de ancho: sin reescalar
        ys, xs = np.nonzero(np.all(hoja == np.array(SV.COL_BUENA, np.uint8), axis=2))
        self.assertLessEqual(abs(xs.min() - 25), 2)
        self.assertLessEqual(abs(xs.max() - 55), 2)
        self.assertLessEqual(abs(ys.min() - 25), 2)
        self.assertLessEqual(abs(ys.max() - 55), 2)

    def test_una_miniatura_ausente_no_rompe(self):
        s = sample(0, [(0.95, (10, 10, 20, 20))])  # sin thumb
        hoja = SV.render_sheet([s], [("SIN_DETECCION",)], [0], 0.5, 0.8, columnas=2, celda=120)
        self.assertEqual(hoja.shape[1], 240)


# ======================================================================
class Base(TD.Base):
    def setUp(self):
        super().setUp()
        self.video = TM.RI.Escenas.p_bg  # 90 frames a 25 fps; el círculo desaparece en los frames 30, 31 y 32

    def escanear(self, sesion=None, **kw):
        msgs = []
        sesion = sesion or TD.SesionBrillo()
        kw.setdefault("cada_s", 0.08)  # a 25 fps: cada 2 frames
        kw.setdefault("salida", os.path.join(self.w, "sal"))
        kw.setdefault("session_factory", lambda r, h: sesion)
        rc = SV.scan(self.video, self.modelo, self.entradas["A"], out=msgs.append, **kw)
        return rc, "\n".join(msgs)

    def filas(self, salida=None):
        with open(os.path.join(salida or os.path.join(self.w, "sal"), "escaneo.csv"), newline="", encoding="utf-8-sig") as fh:
            return list(csv.DictReader(fh))

    def resumen(self, salida=None):
        with open(os.path.join(salida or os.path.join(self.w, "sal"), "resumen.json"), encoding="utf-8") as fh:
            return json.load(fh)


class EscaneoDeUnVideoTests(Base):
    def test_recorre_el_video_con_el_paso_pedido_y_numeros_de_frame_exactos(self):
        rc, texto = self.escanear()
        self.assertEqual(rc, 0, msg=texto)
        f = self.filas()
        self.assertEqual([int(x["frame"]) for x in f], list(range(0, 90, 2)))  # cada 2 frames, desde el 0, hasta el último
        self.assertEqual([float(x["tiempo_s"]) for x in f[:3]], [0.0, 0.08, 0.16])  # frame / fps
        self.assertIn("cada 2 frames (45 imágenes)", texto)
        self.assertIn("25.0 fps", texto)

    def test_marca_los_frames_donde_el_circulo_no_esta(self):
        rc, texto = self.escanear()
        f = {int(x["frame"]): x for x in self.filas()}
        self.assertIn("SIN_DETECCION", f[30]["marcas"])
        self.assertIn("SIN_DETECCION", f[32]["marcas"])
        for k in (28, 34):
            self.assertNotIn("SIN_DETECCION", f[k]["marcas"])
        r = self.resumen()
        self.assertEqual(r["marcas"]["SIN_DETECCION"], 2)
        self.assertEqual(r["con_deteccion"], 43)
        self.assertEqual(r["tramos_sin_deteccion"], [{"desde_s": 30 / 25, "hasta_s": 32 / 25, "imagenes": 2}])
        self.assertIn("Sin detección entre 0:01 y 0:01 (2 imágenes seguidas)", texto)

    def test_la_posicion_cae_sobre_el_circulo(self):
        _, truth = BG.make_frames(90, lost={30, 31, 32})
        self.escanear()
        for x in self.filas():
            f = int(x["frame"])
            if truth[f] is not None and x["cx"]:
                self.assertLess(np.hypot(float(x["cx"]) - truth[f][0], float(x["cy"]) - truth[f][1]), 2.5, msg=f"frame {f}")

    def test_escribe_los_tres_archivos(self):
        rc, texto = self.escanear()
        salida = os.path.join(self.w, "sal")
        self.assertEqual(sorted(os.listdir(salida)), ["casos_dificiles.png", "escaneo.csv", "resumen.json"])
        r = self.resumen()
        self.assertEqual((r["video"], r["modelo"], r["imagenes"], r["cada_s"], r["interrumpido"]), ("bg.avi", "small.onnx", 45, 0.08, False))
        self.assertGreater(r["tiempo_total_s"], 0)
        hoja = cv2.imread(os.path.join(salida, "casos_dificiles.png"))
        self.assertIsNotNone(hoja)
        self.assertEqual(hoja.shape[1], 5 * 320)
        self.assertIn("Casos difíciles", texto)

    def test_hoja_con_pocas_columnas_y_celdas_chicas(self):
        self.escanear(columnas=2, celda=100)
        self.assertEqual(cv2.imread(os.path.join(self.w, "sal", "casos_dificiles.png")).shape[1], 200)

    def test_sin_hoja(self):
        self.escanear(hoja=0)
        self.assertFalse(os.path.exists(os.path.join(self.w, "sal", "casos_dificiles.png")))

    def test_sin_casos_dificiles_se_dice(self):
        with mock.patch.object(SV, "select_cases", return_value=[]):
            _, texto = self.escanear()
        self.assertIn("No hay casos difíciles que mostrar", texto)
        self.assertFalse(os.path.exists(os.path.join(self.w, "sal", "casos_dificiles.png")))

    def test_la_hoja_tiene_a_lo_mas_los_casos_pedidos(self):
        self.escanear(hoja=3, columnas=3, celda=100)
        hoja = cv2.imread(os.path.join(self.w, "sal", "casos_dificiles.png"))
        # 3 casos en 3 columnas = una sola fila; el ROI mide 260 x 190, así que cada celda de 100 de ancho mide round(190 * 100 / 260) = 73 de alto
        self.assertEqual(hoja.shape, (73, 300, 3))

    def test_desde_y_hasta(self):
        rc, texto = self.escanear(desde_s=0.4, hasta_s=1.2)  # frames 10 a 30
        self.assertEqual(rc, 0, msg=texto)
        self.assertEqual([int(x["frame"]) for x in self.filas()], list(range(10, 31, 2)))

    def test_el_final_se_limita_al_video(self):
        self.escanear(hasta_s=999.0)
        self.assertEqual(int(self.filas()[-1]["frame"]), 88)
        self.assertEqual(len(self.filas()), 45)
        _, texto = self.escanear(hasta_s=999.0, salida=os.path.join(self.w, "sal2"))
        self.assertIn("(45 imágenes)", texto)  # el conteo anunciado también se limita al video, no a los 999 s pedidos

    def test_el_paso_se_redondea_no_se_trunca(self):
        self.escanear(cada_s=0.07)  # 25 fps * 0.07 = 1.75 frames: se redondea a 2 (truncado daría 1)
        self.assertEqual([int(x["frame"]) for x in self.filas()][:4], [0, 2, 4, 6])
        self.escanear(cada_s=0.001, salida=os.path.join(self.w, "sal2"))  # nunca menos de 1 frame
        self.assertEqual([int(x["frame"]) for x in self.filas(os.path.join(self.w, "sal2"))][:3], [0, 1, 2])

    def test_con_un_paso_mayor_que_el_limite_del_lector_se_fuerza_la_lectura_sin_saltos(self):
        """El lector busca con saltos (inexactos en un mp4) cuando el paso pasa de 30 frames. Aquí se sube su límite al paso pedido.
        En un AVI de MJPG la búsqueda es exacta y no lo distingue, así que se comprueba el límite directamente."""
        limites = []

        class Espia(SV.VideoReader):
            def iter_frames(self, *a, **k):
                limites.append(self.skip_limit)
                return super().iter_frames(*a, **k)
        with mock.patch.object(SV, "VideoReader", Espia):
            self.escanear(cada_s=1.6)  # paso de 40 frames
            self.escanear(cada_s=0.08, salida=os.path.join(self.w, "sal2"))  # paso de 2: el límite por omisión (30) no se toca
        self.assertEqual(limites, [40, 30])

    def test_las_cajas_de_la_hoja_caen_sobre_el_circulo_de_la_miniatura(self):
        """De punta a punta: la miniatura (recorte reducido) y la caja (coordenadas del frame -> miniatura) deben coincidir."""
        capturado = {}
        real = SV.render_sheet

        def espia(muestras, marcas, indices, *a, **k):
            capturado["muestras"] = muestras
            return real(muestras, marcas, indices, *a, **k)
        with mock.patch.object(SV, "render_sheet", espia):
            self.escanear(celda=320)
        con_caja = [s for s in capturado["muestras"] if s.dets][:6]
        self.assertGreaterEqual(len(con_caja), 3)
        for s in con_caja:
            hoja = real([s], [()], [0], 0.5, 0.8, columnas=1, celda=320)
            verde = np.all(hoja == np.array(SV.COL_BUENA, np.uint8), axis=2)
            claro = np.all(hoja > 200, axis=2) & ~verde  # el círculo blanco de la miniatura
            claro[:22] = False  # sin las franjas donde la hoja escribe el texto (frame y marcas), que también son blancas
            claro[-18:] = False
            ys, xs = np.nonzero(verde)
            cy, cx = np.nonzero(claro)
            self.assertGreater(len(cx), 50, msg=f"frame {s.idx}: no se ve el círculo")
            self.assertLess(abs((xs.min() + xs.max()) / 2 - cx.mean()), 6, msg=f"frame {s.idx}")
            self.assertLess(abs((ys.min() + ys.max()) / 2 - cy.mean()), 6, msg=f"frame {s.idx}")

    def test_un_paso_grande_sigue_dando_frames_exactos(self):
        """Pasos de más de 30 frames: el lector buscaría con saltos inexactos; la herramienta fuerza la lectura sin saltos."""
        rc, _ = self.escanear(cada_s=1.2)  # a 25 fps: cada 30 frames
        self.assertEqual([int(x["frame"]) for x in self.filas()], [0, 30, 60])
        self.escanear(cada_s=1.6, salida=os.path.join(self.w, "sal2"))  # cada 40 frames: mayor que el límite de 30
        self.assertEqual([int(x["frame"]) for x in self.filas(os.path.join(self.w, "sal2"))], [0, 40, 80])

    def test_progreso_y_ritmo(self):
        _, texto = self.escanear()
        self.assertIn("Ritmo: ~", texto)
        self.assertIn("faltan unos", texto)
        self.assertIn("Ctrl+C", texto)
        self.assertIn("imágenes (", texto)  # avances del tipo «18/45 imágenes (40 %)»

    def test_las_marcas_de_varias_detecciones(self):
        _, texto = self.escanear(TD.SesionDos(hw=(96, 96)))  # siempre dos candidatos con confianza suficiente, lejos entre sí
        self.assertEqual(self.resumen()["marcas"]["VARIAS"], 45)
        self.assertEqual({x["n_candidatos"] for x in self.filas()}, {"2"})
        self.assertEqual({x["confianza_2"] for x in self.filas()}, {"0.731"})  # la segunda mejor: sigmoide(1.0); la mejor es sigmoide(4.0) = 0.982

    def test_la_confianza_baja_se_marca(self):
        self.escanear(TD.SesionBrillo(logit=1.0))  # sigmoide(1) = 0.73 < 0.8
        r = self.resumen()
        self.assertEqual(r["marcas"]["CONFIANZA_BAJA"], 43)
        self.escanear(TD.SesionBrillo(logit=1.0), confianza=0.7, salida=os.path.join(self.w, "sal2"))
        self.assertEqual(self.resumen(os.path.join(self.w, "sal2"))["marcas"]["CONFIANZA_BAJA"], 0)

    def test_los_umbrales_llegan_al_detector(self):
        _, _ = self.escanear(TD.SesionBrillo(logit=0.0), umbral_bajo=0.1)  # confianza 0.5: está por debajo del umbral (estricto)
        self.assertEqual(self.resumen()["con_deteccion"], 0)
        filas = self.filas()
        self.assertTrue(all(x["confianza"] == "0.500" for x in filas if x["n_candidatos"] != "0"))  # el candidato débil se ve
        self.escanear(TD.SesionBrillo(logit=0.0), umbral=0.4, umbral_bajo=0.1, salida=os.path.join(self.w, "sal2"))
        self.assertEqual(self.resumen(os.path.join(self.w, "sal2"))["con_deteccion"], 43)

    def test_con_el_umbral_bajo_muy_alto_no_se_ven_los_dudosos(self):
        self.escanear(TD.SesionBrillo(logit=0.0), umbral=0.5, umbral_bajo=0.5)
        self.assertEqual({x["n_candidatos"] for x in self.filas()}, {"0"})  # el candidato de 0.5 no alcanza ni el umbral bajo


class RoiYMargenTests(Base):
    def margen(self, sesion, **kw):
        with mock.patch.object(SV, "crop_rect", wraps=SV.crop_rect) as espia:
            rc, texto = self.escanear(sesion, **kw)
        return rc, texto, {c.args[1] for c in espia.call_args_list}

    def test_el_margen_se_lee_de_las_notas_y_lo_pasado_manda(self):
        rc, texto, m = self.margen(TD.SesionBrillo(notas={"margen_roi_px": 10, "recorte_roi": True}))
        self.assertEqual((rc, m), (0, {10}))
        self.assertIn("Margen del ROI: 10 px (leído de las notas del modelo)", texto)
        _, texto, m = self.margen(TD.SesionBrillo(notas={"margen_roi_px": 10}), margen_roi=0)
        self.assertEqual(m, {0})
        self.assertIn("indicado con --margen-roi", texto)
        self.assertEqual(self.resumen()["margen_roi_px"], 0)

    def test_sin_recorte(self):
        rc, texto = SV.scan(self.video, self.modelo, None, cada_s=0.08, sin_recorte=True, salida=os.path.join(self.w, "s3"), out=lambda m: None,
                            session_factory=lambda r, h: TD.SesionBrillo()), None
        self.assertEqual(rc, 0)
        self.assertEqual(self.resumen(os.path.join(self.w, "s3"))["con_deteccion"], 43)

    def test_el_roi_de_otro_video_se_rechaza_por_la_huella(self):
        msgs = []
        rc = SV.scan(self.video, self.modelo, self.entradas["B"], cada_s=0.08, out=msgs.append, session_factory=lambda r, h: TD.SesionBrillo())
        self.assertEqual(rc, 2)
        self.assertIn("es de otro video", "\n".join(msgs))
        self.assertIn("la huella no coincide", "\n".join(msgs))

    def test_sin_huella_en_la_corrida_se_compara_por_nombre(self):
        copia = os.path.join(self.w, "entradas_sin_huella")
        shutil.copytree(self.entradas["A"], copia)
        d = leer_json(os.path.join(copia, "inputs.json"))
        d["video"].pop("sha256_extremos")
        with open(os.path.join(copia, "inputs.json"), "w", encoding="utf-8") as fh:
            json.dump(d, fh)
        msgs = []
        rc = SV.scan(self.video, self.modelo, copia, cada_s=0.08, salida=os.path.join(self.w, "s4"), out=msgs.append, session_factory=lambda r, h: TD.SesionBrillo())
        self.assertEqual(rc, 0, msg="\n".join(msgs))  # mismo nombre de archivo: se acepta
        d["video"]["path"] = "/otra/carpeta/otro_video.avi"
        with open(os.path.join(copia, "inputs.json"), "w", encoding="utf-8") as fh:
            json.dump(d, fh)
        msgs.clear()
        rc = SV.scan(self.video, self.modelo, copia, cada_s=0.08, out=msgs.append, session_factory=lambda r, h: TD.SesionBrillo())
        self.assertEqual(rc, 2)
        self.assertIn("es de otro_video.avi, no de bg.avi", "\n".join(msgs))

    def test_una_corrida_sin_roi_o_ilegible(self):
        malo = os.path.join(self.w, "entradas_sin_roi")
        shutil.copytree(self.entradas["A"], malo)
        d = leer_json(os.path.join(malo, "inputs.json"))
        del d["roi_box"]
        with open(os.path.join(malo, "inputs.json"), "w", encoding="utf-8") as fh:
            json.dump(d, fh)
        msgs = []
        self.assertEqual(SV.scan(self.video, self.modelo, malo, out=msgs.append, session_factory=lambda r, h: TD.SesionBrillo()), 2)
        self.assertIn("no trae roi_box", "\n".join(msgs))
        msgs.clear()
        self.assertEqual(SV.scan(self.video, self.modelo, self.w, out=msgs.append, session_factory=lambda r, h: TD.SesionBrillo()), 2)

    def test_aviso_de_modelo_entrenado_con_frames_completos(self):
        _, texto = self.escanear(TD.SesionBrillo(notas={"recorte_roi": False}))
        self.assertIn("se entrenó con frames completos", texto)

    def test_la_carpeta_por_omision_esta_junto_a_la_de_la_corrida(self):
        from datetime import datetime
        msgs = []
        rc = SV.scan(self.video, self.modelo, self.entradas["A"], cada_s=0.08, out=msgs.append, session_factory=lambda r, h: TD.SesionBrillo(), now=lambda: datetime(2026, 11, 10, 19, 5, 0))
        esperada = os.path.join(os.path.dirname(os.path.abspath(self.entradas["A"])), "escaneo_20261110_190500")  # dentro de la carpeta del experimento
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(os.path.join(esperada, "resumen.json")))
        self.assertIn(esperada, "\n".join(msgs))
        shutil.rmtree(esperada)


class InterrupcionYErroresTests(Base):
    def test_ctrl_c_analiza_lo_que_se_llevaba(self):
        class Interrumpible(TD.SesionBrillo):
            def run(self, nombres, feed):
                if self.llamadas == 9:
                    raise KeyboardInterrupt
                return super().run(nombres, feed)
        rc, texto = self.escanear(Interrumpible())
        self.assertEqual(rc, 0, msg=texto)
        self.assertIn("Interrumpido: se analizan las 9 imágenes", texto)
        r = self.resumen()
        self.assertEqual((r["imagenes"], r["interrumpido"]), (9, True))
        self.assertEqual(len(self.filas()), 9)

    def test_ctrl_c_antes_de_la_primera_imagen(self):
        class Interrumpible(TD.SesionBrillo):
            def run(self, nombres, feed):
                raise KeyboardInterrupt
        rc, texto = self.escanear(Interrumpible())
        self.assertEqual(rc, 2)
        self.assertIn("No se leyó ninguna imagen", texto)

    def test_un_modelo_de_entrada_dinamica_se_explica(self):
        rc, texto = self.escanear(TD.SesionBrillo(forma=[1, 3, "alto", "ancho"]))
        self.assertEqual(rc, 2)
        self.assertIn("El modelo no se puede usar así", texto)
        self.assertIn("tamaño fijo", texto)

    def test_un_modelo_que_no_abre(self):
        def falla(r, h):
            raise RuntimeError("archivo dañado")
        rc, texto = self.escanear(session_factory=falla)
        self.assertEqual(rc, 2)
        self.assertIn("No se pudo abrir el modelo", texto)
        self.assertIn("archivo dañado", texto)

    def test_un_rango_fuera_del_video(self):
        rc, texto = self.escanear(desde_s=100.0)
        self.assertEqual(rc, 2)
        self.assertIn("No se leyó ninguna imagen", texto)

    def test_el_roi_fuera_del_frame(self):
        copia = os.path.join(self.w, "entradas_fuera")
        shutil.copytree(self.entradas["A"], copia)
        d = leer_json(os.path.join(copia, "inputs.json"))
        d["roi_box"] = [5000, 5000, 100, 100]
        with open(os.path.join(copia, "inputs.json"), "w", encoding="utf-8") as fh:
            json.dump(d, fh)
        msgs = []
        rc = SV.scan(self.video, self.modelo, copia, out=msgs.append, session_factory=lambda r, h: TD.SesionBrillo())
        self.assertEqual(rc, 2)
        self.assertIn("queda fuera del frame", "\n".join(msgs))

    def test_argumentos_invalidos(self):
        casos = [({"cada_s": 0}, "--cada-s"), ({"cada_s": -1}, "--cada-s"), ({"desde_s": -1}, "--cada-s"), ({"desde_s": 5, "hasta_s": 5}, "--cada-s"),
                 ({"umbral_bajo": 0.0}, "umbrales"), ({"umbral": 1.0}, "umbrales"), ({"umbral_bajo": 0.6, "umbral": 0.5}, "umbrales"),
                 ({"confianza": 0.4}, "umbrales"), ({"confianza": 1.1}, "umbrales"), ({"margen_roi": -1}, "--margen-roi"),
                 ({"columnas": 0}, "--columnas"), ({"celda": 10}, "--columnas"), ({"hoja": -1}, "--columnas")]
        for kw, texto in casos:
            rc, out = self.escanear(**kw)
            self.assertEqual(rc, 2, msg=str(kw))
            self.assertIn(texto, out, msg=str(kw))

    def test_archivos_que_faltan(self):
        msgs = []
        fabrica = lambda r, h: TD.SesionBrillo()  # noqa: E731
        self.assertEqual(SV.scan(os.path.join(self.w, "no.mp4"), self.modelo, self.entradas["A"], out=msgs.append, session_factory=fabrica), 2)
        self.assertEqual(SV.scan(self.video, os.path.join(self.w, "no.onnx"), self.entradas["A"], out=msgs.append, session_factory=fabrica), 2)
        self.assertEqual(SV.scan(self.video, self.modelo, None, out=msgs.append, session_factory=fabrica), 2)
        texto = "\n".join(msgs)
        self.assertIn("No existe el video", texto)
        self.assertIn("No existe el modelo", texto)
        self.assertIn("--sin-recorte", texto)

    def test_un_video_que_no_es_un_video(self):
        falso = os.path.join(self.w, "falso.mp4")
        with open(falso, "w") as fh:
            fh.write("no soy un video")
        msgs = []
        rc = SV.scan(falso, self.modelo, None, sin_recorte=True, out=msgs.append, session_factory=lambda r, h: TD.SesionBrillo())
        self.assertEqual(rc, 2)
        self.assertIn("No se pudo abrir el video", "\n".join(msgs))

    def test_cli(self):
        salida = os.path.join(self.w, "cli")
        rc = SV.main([self.video, "--modelo", self.modelo, "--entradas", self.entradas["A"], "--cada-s", "0.08", "--desde-s", "0.4", "--hasta-s", "1.2",
                      "--umbral", "0.4", "--umbral-bajo", "0.1", "--confianza", "0.9", "--margen-roi", "3", "--hilos", "2", "--salida", salida,
                      "--hoja", "6", "--columnas", "3", "--celda", "120"], session_factory=lambda r, h: TD.SesionBrillo())
        self.assertEqual(rc, 0)
        r = self.resumen(salida)
        self.assertEqual((r["umbral"], r["confianza"], r["margen_roi_px"], r["hilos"], r["imagenes"]), (0.4, 0.9, 3, 2, 11))
        with self.assertRaises(SystemExit) as cm:  # falta --modelo: argparse sale con 2
            SV.main([self.video], session_factory=lambda r, h: TD.SesionBrillo())
        self.assertEqual(cm.exception.code, 2)


class MargenCompartidoTests(unittest.TestCase):
    """La regla del margen es una sola, compartida por eval_detector.py y scan_video.py."""

    def test_resolve_margin(self):
        self.assertEqual(ED.resolve_margin({"margen_roi_px": 10}, None, False), (10, "leído de las notas del modelo"))
        self.assertEqual(ED.resolve_margin({"margen_roi_px": 10}, 0, False), (0, "indicado con --margen-roi"))
        self.assertEqual(ED.resolve_margin({"margen_roi_px": 10}, 5, True), (0, "no aplica (frame completo)"))
        self.assertEqual(ED.resolve_margin({}, None, False), (0, "por omisión (el modelo no trae notas)"))
        self.assertEqual(ED.resolve_margin({"margen_roi_px": "diez"}, None, False)[0], 0)
        self.assertEqual(ED.resolve_margin({"margen_roi_px": None}, None, False)[0], 0)
        self.assertEqual(ED.resolve_margin({}, 7, False), (7, "indicado con --margen-roi"))

    def test_las_dos_herramientas_usan_la_misma_funcion(self):
        self.assertIs(SV.resolve_margin, ED.resolve_margin)


if __name__ == "__main__":
    unittest.main()