"""Pruebas de learning/review.py, tools/review_boxes.py y de cómo tools/build_training_set.py aplica revision.json.

Ejecutar desde la raíz del proyecto:
    python -m pytest tests/test_review_boxes.py -v

La ventana se prueba sin pantalla: se le mandan teclas y eventos del ratón a ``Reviewer`` y se mira lo que dibuja ``render()``.
"""

import importlib
import json
import os
import tempfile
import unittest

import cv2
import numpy as np

from learning import review as R
from tools import build_training_set as B
from tools import review_boxes as RB


def _imp(nombre):
    try:
        return importlib.import_module(f"tests.{nombre}")
    except ImportError:
        return importlib.import_module(nombre)


TB = _imp("test_build_training_set")
FW, FH, ROI = TB.FW, TB.FH, TB.ROI
CRUDO = (80, 30, 320, 220)  # recorte del ROI con 20 px de margen: 240 x 190, con zoom 3.0 -> 720 x 570


def ventana(fx, fy, margen=20, bar=RB.BAR_H, escala=3.0, origen=(80, 30)):
    """Coordenadas de la ventana que corresponden al punto (fx, fy) del frame."""
    return (fx - origen[0]) * escala, bar + (fy - origen[1]) * escala


def arrastrar(rv, p0, p1):
    x0, y0 = ventana(*p0)
    x1, y1 = ventana(*p1)
    rv.on_mouse(cv2.EVENT_LBUTTONDOWN, x0, y0)
    rv.on_mouse(cv2.EVENT_MOUSEMOVE, (x0 + x1) / 2, (y0 + y1) / 2)
    rv.on_mouse(cv2.EVENT_LBUTTONUP, x1, y1)


def cuantos(img, color):
    return int(np.all(img == np.array(color, np.uint8), axis=2).sum())


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = self.tmp.name
        self.ds = TB.hacer_dataset(self.t, "exp", "20261101_100000", "rata", "a" * 64, n_train=6, n_val=2)
        self.items, self.size = RB.load_items(self.ds)

    def tearDown(self):
        self.tmp.cleanup()

    def reviewer(self, **kw):
        kw.setdefault("roi_box", ROI)
        return RB.Reviewer(self.ds, self.items, self.size, **kw)

    def disco(self):
        return R.load_revision(self.ds)


class FormatoYEstadisticasTests(Base):
    def test_iou(self):
        self.assertEqual(R.box_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(R.box_iou((0, 0, 10, 10), (20, 20, 5, 5)), 0.0)
        self.assertEqual(R.box_iou((0, 0, 10, 10), (10, 0, 10, 10)), 0.0)  # solo se tocan
        self.assertAlmostEqual(R.box_iou((0, 0, 10, 10), (5, 0, 10, 10)), 50 / 150)
        self.assertAlmostEqual(R.box_iou((0, 0, 10, 10), (2, 2, 4, 4)), 16 / 100)  # una dentro de la otra

    def test_percentil(self):
        self.assertEqual(R._percentile([5.0], 0.1), 5.0)
        self.assertAlmostEqual(R._percentile([0.0, 10.0], 0.1), 1.0)
        self.assertAlmostEqual(R._percentile([1, 2, 3, 4, 5], 0.5), 3.0)

    def rev_de_prueba(self):
        rev = R.new_revision()
        im = rev["imagenes"]
        im["train/a"] = {"estado": "corregida", "original": [100, 100, 50, 40], "final": [90, 100, 70, 40]}  # más ancha
        im["train/b"] = {"estado": "corregida", "original": [200, 200, 60, 60], "final": [210, 200, 60, 60]}  # movida 10 px
        for k in "cde":
            im[f"train/{k}"] = {"estado": "aceptada", "original": [1, 1, 5, 5], "final": [1, 1, 5, 5]}
        im["val/f"] = {"estado": "descartada", "original": [1, 1, 5, 5], "final": None}
        im["val/g"] = {"estado": "dudosa", "original": None, "final": None}
        return rev

    def test_estadisticas_a_mano(self):
        st = R.review_stats(self.rev_de_prueba(), total=12)
        self.assertEqual((st["aceptada"], st["corregida"], st["descartada"], st["dudosa"]), (3, 2, 1, 1))
        self.assertEqual(st["pendientes"], 12 - 7)
        self.assertEqual(st["revisadas"], 5)
        self.assertAlmostEqual(st["pct_corregidas"], 40.0)  # 2 de 5 revisadas
        self.assertEqual(st["n_comparadas"], 2)
        # a: intersección 50x40=2000, unión 2000+2800-2000=2800 -> 0.714286 | b: 50x60=3000, unión 3600+3600-3000=4200 -> 0.714286
        self.assertAlmostEqual(st["iou_mediana"], 5 / 7, places=6)
        self.assertAlmostEqual(st["ancho_razon_mediana"], 1.2)  # (1.4 + 1.0) / 2
        self.assertAlmostEqual(st["alto_razon_mediana"], 1.0)
        self.assertAlmostEqual(st["area_razon_mediana"], 1.2)
        self.assertAlmostEqual(st["desplazamiento_centro_mediana_px"], 5.0)  # (0 + 10) / 2

    def test_estadisticas_sin_correcciones_o_sin_revision(self):
        st = R.review_stats(None, total=5)
        self.assertEqual((st["pendientes"], st["revisadas"], st["pct_corregidas"], st["iou_mediana"]), (5, 0, None, None))
        rev = R.new_revision()
        rev["imagenes"]["train/a"] = {"estado": "aceptada", "original": [1, 1, 5, 5], "final": [1, 1, 5, 5]}
        st = R.review_stats(rev, total=3)
        self.assertEqual(st["pct_corregidas"], 0.0)
        self.assertIsNone(st["iou_mediana"])
        self.assertEqual(R.review_stats(rev)["pendientes"], None)  # sin total no se puede decir

    def test_una_corregida_sin_caja_original_no_entra_a_la_comparacion(self):
        rev = R.new_revision()
        rev["imagenes"]["train/a"] = {"estado": "corregida", "original": None, "final": [1, 1, 5, 5]}
        st = R.review_stats(rev, total=1)
        self.assertEqual((st["corregida"], st["n_comparadas"], st["iou_mediana"]), (1, 0, None))

    def test_texto_del_resumen(self):
        lineas = R.format_stats(R.review_stats(self.rev_de_prueba(), total=12))
        texto = "\n".join(lineas)
        self.assertIn("12 imágenes", texto)
        self.assertIn("pendientes 5", texto)
        self.assertIn("2 (40 %) necesitaron corrección", texto)
        self.assertIn("IoU mediano", texto)
        self.assertIn("1.20 veces el ancho", texto)
        self.assertIn("5.0 px", texto)
        sin = "\n".join(R.format_stats(R.review_stats(R.new_revision(), total=3)))
        self.assertNotIn("necesitaron", sin)
        self.assertNotIn("IoU", sin)

    def test_ida_y_vuelta_atomica(self):
        self.assertIsNone(R.load_revision(self.ds))
        rev = self.rev_de_prueba()
        R.save_revision(self.ds, rev)
        self.assertFalse(os.path.exists(R.revision_path(self.ds) + ".tmp"))  # no queda el archivo temporal
        vuelta = R.load_revision(self.ds)
        self.assertEqual(vuelta["imagenes"], rev["imagenes"])
        self.assertEqual(vuelta["schema"], 1)

    def escribir(self, contenido):
        with open(R.revision_path(self.ds), "w", encoding="utf-8") as fh:
            fh.write(contenido if isinstance(contenido, str) else json.dumps(contenido))

    def test_un_archivo_danado_se_rechaza_sin_ignorarlo(self):
        for malo in ("no es json {", "[]", '{"schema": 2, "imagenes": {}}', '{"schema": 1}', '{"schema": 1, "imagenes": []}'):
            self.escribir(malo)
            with self.assertRaises(ValueError, msg=malo):
                R.load_revision(self.ds)

    def test_entradas_invalidas(self):
        base = {"schema": 1, "imagenes": {}}
        casos = [
            {"estado": "rara", "original": None, "final": None},
            {"estado": "aceptada", "original": [1, 1, 5, 5], "final": None},  # aceptada sin caja final
            {"estado": "corregida", "original": [1, 1, 5, 5], "final": [1, 1, 0, 5]},  # ancho 0
            {"estado": "corregida", "original": [1, 1, 5, 5], "final": [1, 1, -3, 5]},
            {"estado": "corregida", "original": [1, 1, 5, 5], "final": [1, 1, 5]},  # faltan datos
            {"estado": "corregida", "original": [1, 1, 5, 5], "final": [1, 1, "5", 5]},
            {"estado": "corregida", "original": [1, 1, 5, 5], "final": [1, 1, True, 5]},  # un booleano no es un número
            {"estado": "descartada", "original": [0, 0, 0, 0], "final": None},  # original degenerada
            {"estado": "descartada", "original": None, "final": [1, 1]},  # final inválida aunque no se use
            "no soy un diccionario",
        ]
        for c in casos:
            self.escribir({**base, "imagenes": {"train/a": c}})
            with self.assertRaises(ValueError, msg=str(c)):
                R.load_revision(self.ds)
        for raro in ("[1, 1, NaN, 5]", "[NaN, 1, 5, 5]", "[1, NaN, 5, 5]", "[1, 1, Infinity, 5]", "[1, 1, 5, Infinity]", "[-Infinity, 1, 5, 5]"):
            self.escribir('{"schema": 1, "imagenes": {"train/a": {"estado": "corregida", "original": null, "final": %s}}}' % raro)
            with self.assertRaises(ValueError, msg=raro):  # NaN e infinito: json los acepta, pero no son una caja
                R.load_revision(self.ds)
        self.escribir({**base, "imagenes": {"train/a": {"estado": "descartada", "original": None, "final": None},
                                            "train/b": {"estado": "dudosa", "original": [1, 1, 5, 5]}}})
        self.assertEqual(len(R.load_revision(self.ds)["imagenes"]), 2)  # descartada/dudosa sin caja final es válido


class CargaTests(Base):
    def test_orden_tamano_y_cajas_originales(self):
        self.assertEqual(self.size, (FW, FH))
        self.assertEqual([it.key for it in self.items], [f"train/rata_train_{k:03d}" for k in range(6)] + [f"val/rata_val_{k:03d}" for k in range(2)])
        x, y, w, h = self.items[2].original
        for got, want in zip((x, y, w, h), (152, 82, 40, 30)):  # el generador de prueba pone la caja en (150+k, 80+k, 40, 30)
            self.assertAlmostEqual(got, want, delta=0.01)

    def test_el_orden_es_por_frame_aunque_val_este_intercalado_con_train(self):
        """El generador reparte train/val por bloques de tiempo: una imagen de val queda ENTRE dos de train. Se revisa en orden de frame."""
        import shutil
        shutil.copyfile(os.path.join(self.ds, "images", "train", "rata_train_002.jpg"), os.path.join(self.ds, "images", "val", "rata_train_002b.jpg"))
        shutil.copyfile(os.path.join(self.ds, "labels", "train", "rata_train_002.txt"), os.path.join(self.ds, "labels", "val", "rata_train_002b.txt"))
        items, _ = RB.load_items(self.ds)
        claves = [it.key for it in items]
        self.assertEqual(claves[2:5], ["train/rata_train_002", "val/rata_train_002b", "train/rata_train_003"])

    def test_etiquetas_faltantes_o_malas_quedan_sin_caja(self):
        os.remove(os.path.join(self.ds, "labels", "train", "rata_train_000.txt"))
        with open(os.path.join(self.ds, "labels", "train", "rata_train_001.txt"), "w") as fh:
            fh.write("0 .5 .5 .1 .1\n0 .2 .2 .1 .1\n")
        with open(os.path.join(self.ds, "labels", "train", "rata_train_002.txt"), "w") as fh:
            fh.write("basura")
        items, _ = RB.load_items(self.ds)
        self.assertEqual([it.original is None for it in items[:4]], [True, True, True, False])

    def test_conjunto_sin_imagenes(self):
        vacio = os.path.join(self.t, "vacio")
        os.makedirs(vacio)
        with self.assertRaises(ValueError):
            RB.load_items(vacio)


class GeometriaTests(Base):
    def test_recorte_zoom_y_tamano_de_la_ventana(self):
        rv = self.reviewer()
        self.assertEqual(rv.rect, CRUDO)
        self.assertEqual(rv.scale, 3.0)
        self.assertEqual(rv.render().shape, (RB.BAR_H + 570 + RB.BAR_H, 720, 3))

    def test_sin_roi_se_muestra_el_frame_completo(self):
        rv = self.reviewer(roi_box=None)
        self.assertEqual(rv.rect, (0, 0, FW, FH))
        self.assertAlmostEqual(rv.scale, min(3.0, 1100 / FW, 760 / FH))

    def test_el_zoom_se_limita_a_la_ventana(self):
        rv = self.reviewer(max_w=300, max_h=300)
        self.assertAlmostEqual(rv.scale, min(300 / 240, 300 / 190))

    def test_ventana_y_frame_son_inversas(self):
        rv = self.reviewer()
        for fx, fy in ((80, 30), (160, 90), (319, 219)):
            wx, wy = ventana(fx, fy)
            self.assertEqual(rv._to_frame(wx, wy), (fx, fy))
            self.assertEqual(rv._to_view(fx, fy), (int(round((fx - 80) * 3)), int(round((fy - 30) * 3))))

    def test_arrastrar_da_la_caja_exacta_en_coordenadas_del_frame(self):
        rv = self.reviewer()
        arrastrar(rv, (160, 90), (200, 120))
        e = rv.entry()
        self.assertEqual(e["estado"], "corregida")
        self.assertEqual(e["final"], [160.0, 90.0, 40.0, 30.0])  # no del recorte: del frame completo
        self.assertEqual(self.disco()["imagenes"]["train/rata_train_000"]["final"], [160.0, 90.0, 40.0, 30.0])

    def test_arrastrar_en_cualquier_sentido(self):
        rv = self.reviewer()
        arrastrar(rv, (200, 120), (160, 90))  # de abajo-derecha a arriba-izquierda
        self.assertEqual(rv.entry()["final"], [160.0, 90.0, 40.0, 30.0])

    def test_la_caja_se_recorta_a_lo_que_se_ve(self):
        rv = self.reviewer()
        x0, y0 = ventana(160, 90)
        rv.on_mouse(cv2.EVENT_LBUTTONDOWN, x0, y0)
        rv.on_mouse(cv2.EVENT_LBUTTONUP, 5000, 5000)  # mucho más allá de la imagen
        self.assertEqual(rv.entry()["final"], [160.0, 90.0, 160.0, 130.0])  # hasta el borde del recorte (320, 220)
        rv2 = self.reviewer()
        rv2.goto(1)
        rv2.on_mouse(cv2.EVENT_LBUTTONDOWN, 10, 40)
        rv2.on_mouse(cv2.EVENT_LBUTTONUP, -500, -500)
        # arrastrar hacia afuera por arriba-izquierda: la caja debe quedar dentro del recorte visible
        L, T, R_, B_ = CRUDO
        x, y, w, h = rv2.entry()["final"]
        self.assertTrue(L <= x and T <= y and x + w <= R_ and y + h <= B_)

    def test_un_clic_o_un_arrastre_minimo_se_ignora(self):
        rv = self.reviewer()
        arrastrar(rv, (160, 90), (161, 91))  # 1 px del frame: demasiado pequeño
        self.assertIn("pequena", rv.msg)
        self.assertIsNone(self.disco())  # no se guardó nada
        self.assertEqual(rv.state[0], "pendiente")

    def test_soltar_sin_haber_pulsado_no_hace_nada(self):
        rv = self.reviewer()
        rv.on_mouse(cv2.EVENT_LBUTTONUP, 300, 300)
        self.assertIsNone(self.disco())


class DecisionesTests(Base):
    def test_espacio_acepta_la_caja_automatica_y_avanza(self):
        rv = self.reviewer()
        self.assertEqual(rv.state[0], "pendiente")
        self.assertIsNone(rv.on_key(32))
        e = self.disco()["imagenes"]["train/rata_train_000"]
        self.assertEqual(e["estado"], "aceptada")
        self.assertEqual(e["final"], e["original"])
        self.assertEqual(rv.i, 1)

    def test_enter_equivale_a_espacio(self):
        for tecla in (13, 10):
            rv = self.reviewer(start_at_pending=False)
            rv.rev["imagenes"].clear()
            rv.goto(0)
            rv.on_key(tecla)
            self.assertEqual(rv.rev["imagenes"]["train/rata_train_000"]["estado"], "aceptada")

    def test_arrastrar_y_luego_espacio_conserva_la_correccion(self):
        rv = self.reviewer()
        arrastrar(rv, (140, 70), (200, 110))
        e = rv.entry()
        self.assertEqual((e["estado"], e["final"]), ("corregida", [140.0, 70.0, 60.0, 40.0]))
        self.assertEqual(e["original"][0], rv.item.original[0])  # se guarda también la caja automática original
        rv.on_key(32)
        self.assertEqual(self.disco()["imagenes"]["train/rata_train_000"]["estado"], "corregida")  # espacio no la pisa
        self.assertEqual(rv.i, 1)

    def test_volver_a_arrastrar_reemplaza_la_correccion(self):
        rv = self.reviewer()
        arrastrar(rv, (140, 70), (200, 110))
        arrastrar(rv, (150, 80), (190, 110))
        self.assertEqual(rv.entry()["final"], [150.0, 80.0, 40.0, 30.0])
        self.assertEqual(len(self.disco()["imagenes"]), 1)

    def test_x_descarta_y_k_marca_dudosa(self):
        rv = self.reviewer()
        rv.on_key(ord("x"))
        rv.on_key(ord("k"))
        im = self.disco()["imagenes"]
        self.assertEqual((im["train/rata_train_000"]["estado"], im["train/rata_train_000"]["final"]), ("descartada", None))
        self.assertEqual(im["train/rata_train_001"]["estado"], "dudosa")
        self.assertEqual(rv.i, 2)

    def test_espacio_sin_caja_automatica_pide_dibujar_o_descartar(self):
        os.remove(os.path.join(self.ds, "labels", "train", "rata_train_000.txt"))
        self.items, self.size = RB.load_items(self.ds)
        rv = self.reviewer()
        rv.on_key(32)
        self.assertIn("no hay caja", rv.msg)
        self.assertEqual(rv.i, 0)  # no avanza
        self.assertIsNone(self.disco())  # y no se inventa una decisión
        arrastrar(rv, (150, 80), (190, 110))  # dibujar sí se puede, aunque no hubiera caja automática
        self.assertEqual(rv.entry()["original"], None)
        self.assertEqual(rv.entry()["estado"], "corregida")
        self.assertIsNotNone(self.disco())  # y el archivo sigue siendo válido

    def test_u_deshace_y_vuelve_a_la_caja_automatica(self):
        rv = self.reviewer()
        arrastrar(rv, (140, 70), (200, 110))
        rv.on_key(ord("u"))
        self.assertIn("deshecha", rv.msg)
        self.assertEqual(rv.state[0], "pendiente")
        self.assertEqual(rv.state[1], rv.item.original)
        self.assertEqual(self.disco()["imagenes"], {})
        rv.on_key(ord("u"))
        self.assertIn("ya estaba pendiente", rv.msg)
        self.assertEqual(rv.i, 0)

    def test_navegacion(self):
        rv = self.reviewer()
        rv.on_key(ord("a"))
        self.assertIn("primera", rv.msg)
        rv.on_key(ord("d"))
        self.assertEqual(rv.i, 1)
        self.assertIsNone(self.disco())  # moverse no decide nada
        rv.on_key(ord("a"))
        self.assertEqual(rv.i, 0)
        rv.goto(7)
        rv.on_key(ord("d"))
        self.assertIn("ultima", rv.msg)
        self.assertEqual(rv.i, 7)
        rv.on_key(32)  # espacio en la última acepta y se queda
        self.assertEqual(rv.i, 7)
        self.assertEqual(rv.state[0], "aceptada")

    def test_n_salta_a_la_proxima_pendiente_dando_la_vuelta(self):
        rv = self.reviewer()
        for k in range(2, 8):  # deja pendientes solo la 0 y la 1
            rv.goto(k)
            rv.on_key(32)
        rv.goto(5)
        rv.on_key(ord("n"))
        self.assertEqual(rv.i, 0)  # dio la vuelta
        rv.on_key(32)
        rv.on_key(32)  # decide 0 y 1
        rv.on_key(ord("n"))
        self.assertIn("no quedan", rv.msg)

    def test_empieza_en_la_primera_pendiente(self):
        rv = self.reviewer()
        for _ in range(3):
            rv.on_key(32)
        rv2 = self.reviewer()
        self.assertEqual(rv2.i, 3)
        self.assertEqual(self.reviewer(start_at_pending=False).i, 0)
        for _ in range(5):
            rv2.on_key(32)
        self.assertEqual(self.reviewer().i, 0)  # si no queda ninguna, empieza al principio

    def test_lo_decidido_se_conserva_entre_sesiones(self):
        rv = self.reviewer()
        rv.on_key(32)
        rv.goto(1)
        arrastrar(rv, (140, 70), (200, 110))
        rv.on_key(32)  # avanza a la 2 (si se pulsara x aquí, descartaría la 1 y reemplazaría la corrección: la última decisión manda)
        rv.on_key(ord("x"))
        rv2 = self.reviewer()
        c = rv2.counts()
        self.assertEqual((c["aceptada"], c["corregida"], c["descartada"], c["pendiente"], c["total"]), (1, 1, 1, 5, 8))

    def test_la_ultima_decision_manda(self):
        rv = self.reviewer()
        arrastrar(rv, (140, 70), (200, 110))
        rv.on_key(ord("x"))  # descartar la imagen que se acababa de corregir
        self.assertEqual(self.disco()["imagenes"]["train/rata_train_000"]["estado"], "descartada")
        self.assertIsNone(self.disco()["imagenes"]["train/rata_train_000"]["final"])

    def test_salir_y_teclas_raras(self):
        rv = self.reviewer()
        self.assertEqual(rv.on_key(27), "quit")
        self.assertEqual(rv.on_key(ord("q")), "quit")
        self.assertEqual(rv.on_key(ord("Q")), "quit")
        self.assertIsNone(rv.on_key(2 ** 40))  # un código imposible no rompe nada
        self.assertIsNone(rv.on_key(0x110000))
        self.assertIsNone(self.disco())

    def test_un_archivo_danado_no_se_sobrescribe(self):
        with open(R.revision_path(self.ds), "w") as fh:
            fh.write("{ roto")
        with self.assertRaises(ValueError):
            self.reviewer()
        with open(R.revision_path(self.ds)) as fh:
            self.assertEqual(fh.read(), "{ roto")


class DibujoTests(Base):
    def test_el_recorte_se_muestra_con_zoom_y_en_su_lugar(self):
        img = self.reviewer().render()
        # el punto (200, 150) del frame cae en la fila BAR_H + (150-30)*3 = 384, columna (200-80)*3 = 360; allí el degradado vale B=127, G=127
        px = img[RB.BAR_H + 360, 360].astype(int)
        self.assertLess(abs(px[0] - 127), 10)
        self.assertLess(abs(px[1] - 127), 10)
        self.assertLess(abs(px[2] - 128), 10)
        self.assertTrue((img[0, 0] == 30).all() and (img[-1, -1] == 30).all())  # las barras van fuera de la imagen

    def test_colores_segun_el_estado(self):
        rv = self.reviewer()
        pendiente = rv.render()
        self.assertGreater(cuantos(pendiente, RB.COL_AUTO), 100)  # amarilla: la automática, por decidir
        self.assertEqual(cuantos(pendiente, RB.COL_OK), 0)
        rv.on_key(32)
        rv.goto(0)
        aceptada = rv.render()
        self.assertGreater(cuantos(aceptada, RB.COL_OK), 100)
        self.assertEqual(cuantos(aceptada, RB.COL_AUTO), 0)
        rv.goto(1)
        arrastrar(rv, (140, 70), (200, 110))
        corregida = rv.render()
        self.assertGreater(cuantos(corregida, RB.COL_OK), 100)  # la nueva, verde
        self.assertGreater(cuantos(corregida, RB.COL_OLD), 50)  # y la automática original, fina y gris, para comparar
        rv.goto(2)
        rv.on_key(ord("x"))
        rv.goto(2)
        descartada = rv.render()
        rojizos = int(((descartada[:, :, 2] > 200) & (descartada[:, :, 0] < 110) & (descartada[:, :, 1] < 110)).sum())
        self.assertGreater(rojizos, 50)  # rótulo rojo (se dibuja con suavizado: se cuentan los píxeles rojizos)
        self.assertEqual(cuantos(descartada, RB.COL_OK) + cuantos(descartada, RB.COL_AUTO), 0)

    def test_caja_en_construccion(self):
        rv = self.reviewer()
        x0, y0 = ventana(160, 90)
        x1, y1 = ventana(220, 130)
        rv.on_mouse(cv2.EVENT_LBUTTONDOWN, x0, y0)
        rv.on_mouse(cv2.EVENT_MOUSEMOVE, x1, y1)
        self.assertGreater(cuantos(rv.render(), RB.COL_NEW), 100)
        self.assertIsNone(self.disco())  # mientras se arrastra no se guarda nada

    def test_ayuda_y_render_estable(self):
        rv = self.reviewer()
        a = rv.render()
        self.assertTrue(np.array_equal(a, rv.render()))  # dibujar no cambia nada
        rv.on_key(ord("h"))
        b = rv.render()
        self.assertFalse(np.array_equal(a, b))
        self.assertTrue(rv.show_help)
        rv.on_key(ord("h"))
        self.assertTrue(np.array_equal(a, rv.render()))

    def test_una_imagen_de_otro_tamano_se_rechaza(self):
        cv2.imwrite(self.items[0].image_path, np.zeros((200, 300, 3), np.uint8))
        with self.assertRaises(OSError):
            self.reviewer().render()

    def test_una_imagen_ilegible(self):
        with open(self.items[0].image_path, "w") as fh:
            fh.write("no soy una imagen")
        with self.assertRaises(OSError):
            self.reviewer().render()


class MainTests(Base):
    def correr(self, args, gui=None):
        msgs, llamadas = [], []

        def falso(rv):
            llamadas.append(rv)
            if gui:
                gui(rv)

        rc = RB.main(args, gui=falso, out=msgs.append)
        return rc, "\n".join(msgs), llamadas

    def test_solo_resumen_no_abre_la_ventana(self):
        rv = self.reviewer()
        rv.on_key(32)
        arrastrar(rv, (140, 70), (200, 110))
        rc, texto, llamadas = self.correr([self.ds, "--solo-resumen"])
        self.assertEqual(rc, 0)
        self.assertEqual(llamadas, [])
        self.assertIn("8 imágenes", texto)
        self.assertIn("aceptadas 1", texto)
        self.assertIn("corregidas 1", texto)
        self.assertIn("IoU mediano", texto)

    def test_con_ventana_se_abre_una_vez_y_al_salir_se_imprime_el_resumen(self):
        def decide(rv):
            rv.on_key(32)
            rv.on_key(ord("x"))

        rc, texto, llamadas = self.correr([self.ds], gui=decide)
        self.assertEqual(rc, 0)
        self.assertEqual(len(llamadas), 1)
        self.assertIsInstance(llamadas[0], RB.Reviewer)
        self.assertIn("aceptadas 1", texto)
        self.assertIn("descartadas 1", texto)

    def test_desde_inicio_y_margen(self):
        self.reviewer().on_key(32)
        rc, _, ll = self.correr([self.ds, "--desde-inicio", "--margen-roi", "0"])
        self.assertEqual(ll[0].i, 0)
        self.assertEqual(ll[0].rect, (100, 50, 300, 200))
        rc, _, ll = self.correr([self.ds])
        self.assertEqual(ll[0].i, 1)  # por omisión, en la primera pendiente

    def test_sin_roi_avisa_y_sigue(self):
        d = TB.hacer_dataset(self.t, "exp2", "20261102_100000", "rata2", "b" * 64, roi_en_info=False)
        rc, texto, ll = self.correr([d])
        self.assertEqual(rc, 0)
        self.assertIn("no se encontró el ROI", texto)
        self.assertEqual(ll[0].rect, (0, 0, FW, FH))

    def test_imagenes_sin_caja_se_avisan(self):
        os.remove(os.path.join(self.ds, "labels", "train", "rata_train_000.txt"))
        _, texto, _ = self.correr([self.ds, "--solo-resumen"])
        self.assertIn("1 imágenes no tienen una caja automática", texto)

    def test_errores_de_uso(self):
        rc, texto, ll = self.correr([self.t])
        self.assertEqual(rc, 2)
        self.assertIn("dataset_info.json", texto)
        self.assertEqual(ll, [])
        vacio = os.path.join(self.t, "dataset_20269999_000000")
        os.makedirs(vacio)
        with open(os.path.join(vacio, "dataset_info.json"), "w") as fh:
            fh.write("{}")
        self.assertEqual(self.correr([vacio])[0], 2)
        with open(R.revision_path(self.ds), "w") as fh:
            fh.write("{ roto")
        rc, texto, ll = self.correr([self.ds])
        self.assertEqual(rc, 2)
        self.assertIn("dañado", texto)
        self.assertEqual(ll, [])


class AplicacionEnElArmadoTests(unittest.TestCase):
    """build_training_set aplica lo decidido en la revisión."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = self.tmp.name
        self.salida = os.path.join(self.t, "entrenamiento")
        self.A = TB.hacer_dataset(self.t, "expA", "20261101_100000", "rata", "a" * 64, n_train=6, n_val=2)
        self.Bd = TB.hacer_dataset(self.t, "expB", "20261102_100000", "ratb", "b" * 64, n_train=3, n_val=1)

    def tearDown(self):
        self.tmp.cleanup()

    def revisar(self, ds, decisiones):
        rev = R.new_revision()
        for clave, (estado, original, final) in decisiones.items():
            rev["imagenes"][clave] = {"estado": estado, "original": original, "final": final}
        R.save_revision(ds, rev)

    def armar(self, train, valid=(), **kw):
        msgs = []
        rc = B.build(train, valid, self.salida, out=msgs.append, **kw)
        return rc, "\n".join(msgs)

    def ficha(self):
        with open(os.path.join(self.salida, "conjunto_info.json"), encoding="utf-8") as fh:
            return json.load(fh)

    def etiqueta(self, split, stem):
        return TB.leer(os.path.join(self.salida, split, "labels", stem + ".txt")).strip()

    def decisiones_mixtas(self):
        o = lambda k: [150.0 + k, 80.0 + k, 40.0, 30.0]  # noqa: E731
        self.revisar(self.A, {
            "train/rata_train_000": ("corregida", o(0), [140.0, 70.0, 60.0, 40.0]),
            "train/rata_train_001": ("aceptada", o(1), o(1)),
            "train/rata_train_002": ("descartada", o(2), None),
            "train/rata_train_003": ("dudosa", o(3), None),
        })

    def test_descartadas_y_dudosas_no_entran_y_la_correccion_si(self):
        self.decisiones_mixtas()
        rc, texto = self.armar([self.A], [self.Bd])
        self.assertEqual(rc, 0)
        self.assertEqual(self.ficha()["imagenes"], {"train": 6, "valid": 4})  # 8 de A menos 2, y 4 de B
        fa = self.ficha()["fuentes"][0]
        self.assertEqual(fa["motivos_de_omision"], {"descartada_en_la_revision": 1, "dudosa_en_la_revision": 1})
        self.assertEqual(fa["omitidas"], 2)
        self.assertEqual(fa["revision"], {"tiene_revision": True, "usadas_aceptadas": 1, "usadas_corregidas": 1, "usadas_sin_revisar": 4})
        listado = os.listdir(os.path.join(self.salida, "train", "images"))
        self.assertFalse(any("train_002" in f or "train_003" in f for f in listado))
        # caja corregida (140,70,60,40) en el frame -> en el recorte del ROI (40,20,60,40) de 200x150
        TB.casi(self, self.etiqueta("train", "rata_aaaaaaaa_rata_train_000"), "0 0.350000 0.266667 0.300000 0.266667")

    def test_la_ficha_cuenta_aceptadas_y_corregidas_por_separado(self):
        o = lambda k: [150.0 + k, 80.0 + k, 40.0, 30.0]  # noqa: E731
        self.revisar(self.A, {"train/rata_train_000": ("corregida", o(0), [140.0, 70.0, 60.0, 40.0]),
                              "train/rata_train_001": ("corregida", o(1), [141.0, 71.0, 60.0, 40.0]),
                              "train/rata_train_002": ("aceptada", o(2), o(2))})  # 2 corregidas y 1 aceptada: asimétrico a propósito
        self.armar([self.A], [self.Bd])
        r = self.ficha()["fuentes"][0]["revision"]
        self.assertEqual((r["usadas_aceptadas"], r["usadas_corregidas"], r["usadas_sin_revisar"]), (1, 2, 5))

    def test_la_caja_aceptada_y_la_pendiente_usan_la_automatica(self):
        self.decisiones_mixtas()
        self.armar([self.A], [self.Bd])
        for k in (1, 4):  # 1 = aceptada, 4 = pendiente: las dos salen con la caja automática (150+k, 80+k, 40, 30) convertida al recorte
            x, y = 150 + k - 100, 80 + k - 50
            esperado = f"0 {(x + 20) / 200:.6f} {(y + 15) / 150:.6f} {40 / 200:.6f} {30 / 150:.6f}"
            TB.casi(self, self.etiqueta("train", f"rata_aaaaaaaa_rata_train_00{k}"), esperado)

    def test_solo_revisadas(self):
        self.revisar(self.A, {
            "train/rata_train_000": ("corregida", [150.0, 80.0, 40.0, 30.0], [140.0, 70.0, 60.0, 40.0]),
            "train/rata_train_001": ("aceptada", [151.0, 81.0, 40.0, 30.0], [151.0, 81.0, 40.0, 30.0]),
            "val/rata_val_000": ("aceptada", [150.0, 80.0, 40.0, 30.0], [150.0, 80.0, 40.0, 30.0]),
        })
        rc, texto = self.armar([self.A], solo_revisadas=True)  # sin --valid: val interno
        self.assertEqual(rc, 0)
        self.assertEqual(self.ficha()["imagenes"], {"train": 2, "valid": 1})
        fa = self.ficha()["fuentes"][0]
        self.assertEqual(fa["motivos_de_omision"], {"sin_revisar": 5})
        self.assertTrue(self.ficha()["solo_revisadas"])
        self.assertIn("Todas las imágenes usadas fueron revisadas", texto)
        self.assertNotIn("AUTOMÁTICAS", texto)  # ya no son etiquetas solo automáticas

    def test_el_tope_por_video_se_reparte_entre_las_utilizables(self):
        A10 = TB.hacer_dataset(self.t, "expC", "20261103_100000", "ratc", "c" * 64, n_train=10, n_val=2)
        self.revisar(A10, {f"train/ratc_train_{k:03d}": ("aceptada", [150.0 + k, 80.0 + k, 40.0, 30.0], [150.0 + k, 80.0 + k, 40.0, 30.0]) for k in range(3)}
                     | {"val/ratc_val_000": ("aceptada", [150.0, 80.0, 40.0, 30.0], [150.0, 80.0, 40.0, 30.0])})
        rc, _ = self.armar([A10], solo_revisadas=True, max_por_video=5)
        self.assertEqual(rc, 0)
        # hay 3 aceptadas de 10: el tope de 5 no debe "gastarse" en pendientes que luego se descartan
        self.assertEqual(self.ficha()["imagenes"]["train"], 3)

    def test_la_revision_usa_la_division_original_aunque_cambie_el_destino(self):
        """Con --valid, las imágenes val/ de un video de entrenamiento van a train/, pero su decisión sigue siendo la de val/."""
        self.revisar(self.A, {"val/rata_val_000": ("descartada", [150.0, 80.0, 40.0, 30.0], None)})
        rc, _ = self.armar([self.A], [self.Bd])
        self.assertEqual(rc, 0)
        self.assertFalse(any("val_000" in f for f in os.listdir(os.path.join(self.salida, "train", "images"))))
        self.assertEqual(self.ficha()["imagenes"]["train"], 7)

    def test_una_caja_corregida_no_necesita_la_etiqueta_original(self):
        os.remove(os.path.join(self.A, "labels", "train", "rata_train_005.txt"))
        os.remove(os.path.join(self.A, "labels", "train", "rata_train_004.txt"))
        self.revisar(self.A, {"train/rata_train_005": ("corregida", None, [150.0, 80.0, 40.0, 30.0])})
        rc, _ = self.armar([self.A], [self.Bd])
        self.assertEqual(rc, 0)
        fa = self.ficha()["fuentes"][0]
        self.assertEqual(fa["motivos_de_omision"], {"sin_etiqueta": 1})  # la 004 sigue sin etiqueta y sin revisión
        self.assertTrue(os.path.exists(os.path.join(self.salida, "train", "labels", "rata_aaaaaaaa_rata_train_005.txt")))

    def test_sin_revision_el_comportamiento_es_el_de_siempre(self):
        rc, texto = self.armar([self.A], [self.Bd])
        self.assertEqual(rc, 0)
        self.assertEqual(self.ficha()["imagenes"], {"train": 8, "valid": 4})
        self.assertFalse(self.ficha()["fuentes"][0]["revision"]["tiene_revision"])
        self.assertIn("AUTOMÁTICAS", texto)
        self.assertIn("12 de 12 imágenes usadas no se revisaron", texto)

    def test_aviso_de_cuantas_quedan_sin_revisar(self):
        self.decisiones_mixtas()
        rc, texto = self.armar([self.A], [self.Bd])
        self.assertIn("8 de 10 imágenes usadas no se revisaron", texto)  # 4 de A (pendientes) + 4 de B
        self.assertIn("AUTOMÁTICAS", texto)

    def test_revision_danada_detiene_el_armado(self):
        with open(R.revision_path(self.A), "w") as fh:
            fh.write("{ roto")
        rc, texto = self.armar([self.A], [self.Bd])
        self.assertEqual(rc, 2)
        self.assertIn("dañado", texto)
        self.assertFalse(os.path.exists(self.salida))

    def test_solo_revisadas_sin_ninguna_revisada_falla_con_motivo(self):
        rc, texto = self.armar([self.A], [self.Bd], solo_revisadas=True)
        self.assertEqual(rc, 2)

    def test_cli(self):
        self.decisiones_mixtas()
        self.assertEqual(B.main(["--train", self.A, "--valid", self.Bd, "--salida", self.salida, "--solo-revisadas"]), 2)  # B sin revisar: sin validación
        self.assertEqual(B.main(["--train", self.A, "--valid", self.Bd, "--salida", self.salida]), 0)

    def test_de_punta_a_punta_con_la_ventana_de_revision(self):
        """Lo que se decide en la ventana (teclas y arrastres) termina, convertido, en las etiquetas de entrenamiento."""
        items, size = RB.load_items(self.A)
        rv = RB.Reviewer(self.A, items, size, ROI)
        arrastrar(rv, (140, 70), (200, 110))  # imagen 0: se corrige a (140, 70, 60, 40)
        rv.on_key(32)  # espacio sobre una ya corregida: solo avanza (a la 1)
        rv.on_key(32)  # la 1 se acepta (y avanza a la 2)
        rv.on_key(ord("x"))  # la 2 se descarta
        rc, _ = self.armar([self.A], [self.Bd])
        self.assertEqual(rc, 0)
        TB.casi(self, self.etiqueta("train", "rata_aaaaaaaa_rata_train_000"), "0 0.350000 0.266667 0.300000 0.266667")
        self.assertFalse(any("train_002" in f for f in os.listdir(os.path.join(self.salida, "train", "images"))))
        self.assertEqual(R.review_stats(rv.rev, total=8)["corregida"], 1)

class ConElGeneradorRealTests(unittest.TestCase):
    """La ventana de revisión y el armado sobre la salida REAL del generador (video sintético con posición conocida)."""

    W, H = TB.IntegracionConElGeneradorRealTests.W, TB.IntegracionConElGeneradorRealTests.H
    ROI = TB.IntegracionConElGeneradorRealTests.ROI

    @classmethod
    def setUpClass(cls):
        TB.IntegracionConElGeneradorRealTests.setUpClass.__func__(cls)  # crea cls.datasets, cls.verdad, cls.videos, cls.tmp

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        for d in self.datasets.values():
            if os.path.exists(R.revision_path(d)):
                os.remove(R.revision_path(d))
        self.salida = os.path.join(self.tmp.name, "salida_rev")
        if os.path.exists(self.salida):
            import shutil
            shutil.rmtree(self.salida)

    def test_las_cajas_automaticas_reales_se_cargan_sobre_el_circulo(self):
        items, size = RB.load_items(self.datasets["a"])
        self.assertEqual(size, (self.W, self.H))
        self.assertGreaterEqual(len(items), 6)
        for it in items:
            self.assertIsNotNone(it.original)
            frame = int(it.key.split("_")[-1])
            x, y, w, h = it.original
            vx, vy = self.verdad["a"][frame]
            self.assertLess(((x + w / 2 - vx) ** 2 + (y + h / 2 - vy) ** 2) ** 0.5, 3.0, msg=it.key)

    def test_la_ventana_dibuja_la_caja_real_con_zoom(self):
        info = B.read_info(self.datasets["a"])
        items, size = RB.load_items(self.datasets["a"])
        roi, _ = B.find_roi(self.datasets["a"], info)
        rv = RB.Reviewer(self.datasets["a"], items, size, roi, margin=5)
        img = rv.render()
        self.assertGreater(cuantos(img, RB.COL_AUTO), 50)  # la caja automática, amarilla, sobre la imagen real
        self.assertGreater(rv.scale, 2.0)  # el recorte es chico: se amplía

    def test_revisar_todo_y_armar_el_conjunto_solo_con_lo_revisado(self):
        for nombre in ("a", "b"):
            items, size = RB.load_items(self.datasets[nombre])
            rv = RB.Reviewer(self.datasets[nombre], items, size, self.ROI)
            for _ in items:
                rv.on_key(32)  # aceptar todas
            self.assertEqual(rv.counts()["pendiente"], 0)
        msgs = []
        rc = B.build([self.datasets["a"]], [self.datasets["b"]], self.salida, solo_revisadas=True, out=msgs.append)
        self.assertEqual(rc, 0)
        self.assertIn("Todas las imágenes usadas fueron revisadas", "\n".join(msgs))
        x0, y0, w0, h0 = self.ROI
        n = 0
        for sp, nombre in (("train", "a"), ("valid", "b")):
            for f in os.listdir(os.path.join(self.salida, sp, "images")):
                frame = int(os.path.splitext(f)[0].split("_")[-1])
                linea = TB.leer(os.path.join(self.salida, sp, "labels", os.path.splitext(f)[0] + ".txt")).strip()
                _, (x, y, w, h) = TB.parse_yolo_label_line(linea, w0, h0)
                vx, vy = self.verdad[nombre][frame]
                self.assertLess(((x + w / 2) - (vx - x0)) ** 2 + ((y + h / 2) - (vy - y0)) ** 2, 9.0)
                n += 1
        self.assertGreaterEqual(n, 6)

    def test_resumen_por_linea_de_comandos_sobre_el_conjunto_real(self):
        msgs = []
        rc = RB.main([self.datasets["a"], "--solo-resumen"], gui=lambda rv: self.fail("no debe abrir la ventana"), out=msgs.append)
        self.assertEqual(rc, 0)
        self.assertIn("pendientes", "\n".join(msgs))


if __name__ == "__main__":
    unittest.main()