"""Pruebas de tools/audit_labels.py (hoja de auditoría de las cajas etiquetadas a mano).

Ejecutar desde la raíz del proyecto:
    python -m pytest tests/test_audit_labels.py -v
"""

import importlib
import os
import tempfile
import unittest

import cv2
import numpy as np

from tools import audit_labels as A


def _imp(nombre):
    try:
        return importlib.import_module(f"tests.{nombre}")
    except ImportError:
        return importlib.import_module(nombre)


TM = _imp("test_build_manual")
CAJA, ALERTA = A.COL_CAJA, A.COL_ALERTA


def escribir(ruta, texto=""):
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write(texto)


def item(frame, w, h, video="v", imagen="", x=100, y=100):
    return A.Item(video, video, frame, imagen, (x, y, w, h), 400, 300)


def pixeles(celda, color):
    """Rectángulo envolvente (x0, y0, x1, y1) de los píxeles de un color exacto en una celda; None si no hay."""
    ys, xs = np.nonzero(np.all(celda == np.array(color, np.uint8), axis=2))
    return None if len(xs) == 0 else (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


# ======================================================================
class EstadisticaTests(unittest.TestCase):
    def test_cuartiles_a_mano(self):
        self.assertEqual(A._cuartiles([1, 2, 3, 4]), (1.75, 3.25))  # pos 0.75 y 2.25 con interpolación lineal
        self.assertEqual(A._cuartiles([5, 5, 5, 5]), (5, 5))
        self.assertEqual(A._cuartiles([10]), (10, 10))
        self.assertEqual(A._cuartiles([4, 1, 3, 2]), (1.75, 3.25))  # no depende del orden de entrada

    def test_limites_de_tukey_a_mano(self):
        lo, hi = A.tukey_bounds([1, 2, 3, 4])  # Q1 1.75, Q3 3.25, IQR 1.5 -> 1.75 - 2.25 y 3.25 + 2.25
        self.assertAlmostEqual(lo, -0.5)
        self.assertAlmostEqual(hi, 5.5)
        self.assertEqual(A.tukey_bounds([10] * 8 + [100]), (10, 10))  # sin dispersión, cualquier cosa distinta es rara
        lo2, hi2 = A.tukey_bounds([1, 2, 3, 4], k=3.0)
        self.assertAlmostEqual(hi2, 3.25 + 3 * 1.5)

    def test_marca_solo_lo_que_se_sale_por_arriba_y_por_abajo(self):
        normales = [item(f, 30, 30) for f in range(10)]  # área 900
        grande, chica = item(100, 90, 90), item(101, 10, 10)
        marcadas = A.flag_outliers(normales + [grande, chica])
        self.assertEqual(sorted(i.frame for i in marcadas), [100, 101])
        self.assertEqual(A.flag_outliers(normales + [grande]), [grande])  # solo por arriba
        self.assertEqual(A.flag_outliers(normales + [chica]), [chica])  # solo por abajo
        self.assertEqual(A.flag_outliers(normales), [])

    def test_con_pocas_cajas_no_se_marca_nada(self):
        siete = [item(f, 30, 30) for f in range(6)] + [item(99, 300, 300)]
        self.assertEqual(len(siete), 7)
        self.assertEqual(A.flag_outliers(siete), [])  # sin base para decir qué es habitual
        self.assertEqual([i.frame for i in A.flag_outliers(siete + [item(98, 30, 30)])], [99])  # con 8 ya sí

    def test_orden(self):
        a, b, c = item(3, 50, 10, "z"), item(1, 20, 20, "a"), item(2, 40, 40, "a")
        self.assertEqual([i.frame for i in A.sort_items([a, b, c], "area")], [1, 3, 2])  # áreas: b=400, a=500, c=1600
        self.assertEqual([i.frame for i in A.sort_items([a, b, c], "frame")], [1, 2, 3])  # por video y luego frame
        self.assertEqual([i.frame for i in A.sort_items([a, b, c], "aspecto")], [1, 2, 3])  # w/h: 1.0, 1.0, 5.0 (empate por frame)
        with self.assertRaises(ValueError):
            A.sort_items([a], "tamaño")

    def test_orden_por_frame_agrupa_por_video_antes_que_por_numero(self):
        z1, a2, a9 = item(1, 30, 30, "z"), item(2, 30, 30, "a"), item(9, 30, 30, "a")
        self.assertEqual([(i.video, i.frame) for i in A.sort_items([z1, a9, a2], "frame")], [("a", 2), ("a", 9), ("z", 1)])  # no [1, 2, 9]

    def test_orden_por_area_y_desempate(self):
        x, y, z = item(5, 30, 20), item(4, 20, 30), item(9, 10, 10)  # áreas 600, 600, 100
        self.assertEqual([i.frame for i in A.sort_items([x, y, z], "area")], [9, 4, 5])  # de menor a mayor, y a igual área por frame

    def test_propiedades_del_item(self):
        i = item(1, 50, 20)
        self.assertEqual((i.area, i.aspecto), (1000, 2.5))


# ======================================================================
class RecorteTests(unittest.TestCase):
    def setUp(self):
        h, w = 100, 200
        self.img = np.zeros((h, w, 3), np.uint8)
        self.img[:, :, 0] = np.arange(w, dtype=np.uint8)[None, :]  # el canal B codifica x (w < 256)
        self.img[:, :, 1] = np.arange(h, dtype=np.uint8)[:, None]  # el G codifica y

    def test_centrado(self):
        crop, x0, y0 = A.extract(self.img, 100, 50, 40)
        self.assertEqual((crop.shape, x0, y0), ((40, 40, 3), 80, 30))
        np.testing.assert_array_equal(crop, self.img[30:70, 80:120])

    def test_cerca_de_un_borde_la_ventana_se_desplaza_y_conserva_su_tamano(self):
        crop, x0, y0 = A.extract(self.img, 5, 5, 40)  # esquina superior izquierda
        self.assertEqual((crop.shape, x0, y0), ((40, 40, 3), 0, 0))
        np.testing.assert_array_equal(crop, self.img[0:40, 0:40])
        crop, x0, y0 = A.extract(self.img, 199, 99, 40)  # esquina inferior derecha
        self.assertEqual((crop.shape, x0, y0), ((40, 40, 3), 160, 60))
        np.testing.assert_array_equal(crop, self.img[60:100, 160:200])

    def test_ventana_mayor_que_la_imagen_se_rellena_de_negro(self):
        crop, x0, y0 = A.extract(self.img, 100, 50, 300)
        self.assertEqual((crop.shape, x0, y0), ((300, 300, 3), 0, 0))
        np.testing.assert_array_equal(crop[:100, :200], self.img)
        self.assertEqual(int(crop[100:].sum()) + int(crop[:, 200:].sum()), 0)  # lo que sobra es negro

    def test_ventana_de_tamano_impar(self):
        crop, x0, y0 = A.extract(self.img, 100, 50, 41)
        self.assertEqual(crop.shape, (41, 41, 3))


# ======================================================================
class HojaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def imagen(self, nombre="i.png", w=400, h=300):
        ruta = os.path.join(self.t, nombre)
        cv2.imwrite(ruta, np.full((h, w, 3), 128, np.uint8))
        return ruta

    def test_tamano_y_posicion_de_las_celdas(self):
        ruta = self.imagen()
        items = [item(f, 40, 30, imagen=ruta, x=150, y=120) for f in range(5)]
        hoja = A.render(items, lado=120, columnas=3, celda=240)
        self.assertEqual(hoja.shape, (2 * 240, 3 * 240, 3))  # 5 cajas en 3 columnas: 2 filas
        for k in range(5):
            r, c = divmod(k, 3)
            self.assertIsNotNone(pixeles(hoja[r * 240:(r + 1) * 240, c * 240:(c + 1) * 240], CAJA), msg=f"celda {k}")
        sobrante = hoja[240:, 2 * 240:]  # la sexta celda no tiene caja: queda el fondo
        self.assertIsNone(pixeles(sobrante, CAJA))
        self.assertTrue((sobrante[10:-10, 10:-10] == 25).all())

    def test_cada_caja_va_en_su_celda_llenando_por_filas(self):
        """Cajas de anchos distintos: la que está en la posición k de la lista debe quedar en la fila k // columnas y la columna k % columnas."""
        ruta = self.imagen()
        anchos = [20, 30, 40, 50, 60]
        items = [item(k, w, 20, imagen=ruta, x=150, y=120) for k, w in enumerate(anchos)]
        hoja = A.render(items, lado=120, columnas=3, celda=240)  # escala 2: el ancho dibujado es el doble
        for k, w in enumerate(anchos):
            r, c = divmod(k, 3)
            x0, _, x1, _ = pixeles(hoja[r * 240:(r + 1) * 240, c * 240:(c + 1) * 240], CAJA)
            self.assertLessEqual(abs((x1 - x0) - 2 * w), 3, msg=f"la celda ({r},{c}) debería tener la caja de la posición {k}")

    def test_la_caja_se_dibuja_centrada_y_a_escala(self):
        ruta = self.imagen()
        hoja = A.render([item(1, 40, 30, imagen=ruta, x=150, y=120)], lado=120, columnas=1, celda=240)  # escala 2
        x0, y0, x1, y1 = pixeles(hoja, CAJA)
        self.assertLessEqual(abs((x1 - x0) - 80), 3)  # 40 px * 2
        self.assertLessEqual(abs((y1 - y0) - 60), 3)  # 30 px * 2
        self.assertLessEqual(abs((x0 + x1) / 2 - 120), 3)  # centrada en la celda
        self.assertLessEqual(abs((y0 + y1) / 2 - 120), 3)

    def test_misma_escala_para_todas(self):
        """Una caja el doble de ancha se dibuja el doble de ancha: así se comparan tamaños de verdad."""
        ruta = self.imagen()
        hoja = A.render([item(1, 40, 30, imagen=ruta, x=150, y=120), item(2, 80, 30, imagen=ruta, x=130, y=120)], lado=160, columnas=2, celda=160)
        a, b = pixeles(hoja[:, :160], CAJA), pixeles(hoja[:, 160:], CAJA)
        self.assertLessEqual(abs((b[2] - b[0]) - 2 * (a[2] - a[0])), 4)
        self.assertLessEqual(abs((b[3] - b[1]) - (a[3] - a[1])), 2)  # y la altura igual

    def test_borde_rojo_para_las_alertas(self):
        ruta = self.imagen()
        normal, rara = item(1, 40, 30, imagen=ruta), item(2, 40, 30, imagen=ruta)
        hoja = A.render([normal, rara], lado=120, columnas=2, celda=240, alertas=[rara])
        self.assertIsNotNone(pixeles(hoja[:, :240], CAJA))
        self.assertIsNone(pixeles(hoja[:, :240], ALERTA))
        self.assertIsNotNone(pixeles(hoja[:, 240:], ALERTA))
        self.assertIsNone(pixeles(hoja[:, 240:], CAJA))

    def test_una_imagen_ilegible_deja_su_celda_vacia(self):
        mala = os.path.join(self.t, "mala.png")
        escribir(mala, "no soy una imagen")
        hoja = A.render([item(1, 40, 30, imagen=mala), item(2, 40, 30, imagen=self.imagen())], lado=120, columnas=2, celda=120)
        self.assertIsNone(pixeles(hoja[:, :120], CAJA))
        self.assertIsNotNone(pixeles(hoja[:, 120:], CAJA))

    def test_la_caja_cerca_del_borde_se_dibuja_donde_esta(self):
        ruta = self.imagen()
        # caja pegada a la esquina superior izquierda: la ventana se desplaza y la caja ya no queda en el centro de la celda
        hoja = A.render([item(1, 40, 30, imagen=ruta, x=0, y=0)], lado=120, columnas=1, celda=240)
        x0, y0, x1, y1 = pixeles(hoja, CAJA)
        self.assertLessEqual(x0, 3)
        self.assertLessEqual(y0, 3)
        self.assertLessEqual(abs((x1 - x0) - 80), 3)

    def test_sin_cajas(self):
        with self.assertRaises(ValueError):
            A.render([], 100, 4, 100)


# ======================================================================
class ConjuntoRealTests(TM.Base):
    def etiquetado(self, cajas=None, claves=("A",), por_video=12, **kw):
        return self.conjunto(claves, por_video=por_video, cajas=cajas, **kw)

    def test_lee_cajas_en_pixeles_del_frame_completo(self):
        val, vt = self.etiquetado()
        items, cuenta = A.collect(val)
        self.assertEqual(cuenta, {"con_caja": 12, "sin_raton": 0, "varias_cajas": 0, "invalidas": 0, "sin_etiqueta": 0})
        for it in items:
            for got, want in zip(it.caja, TM.caja(it.frame)):
                self.assertAlmostEqual(got, want, delta=0.01)
            self.assertEqual((it.ancho, it.alto), (320, 240))
            self.assertTrue(os.path.exists(it.imagen))
        self.assertEqual([i.frame for i in items], sorted(vt[0]["frames"]))

    def test_cuenta_cada_clase_de_etiqueta(self):
        val, vt = self.etiquetado()
        v = vt[0]
        f = sorted(v["frames"])
        lab = lambda k: os.path.join(val, "labels", f"{v['id']}_f{f[k]:06d}.txt")  # noqa: E731
        os.remove(lab(0))  # dudosa o pendiente
        escribir(lab(1))  # sin ratón
        escribir(lab(2), "0 .5 .5 .1 .1\n0 .3 .3 .1 .1\n")  # dos cajas
        escribir(lab(3), "basura")
        escribir(lab(4), "1 0.5 0.5 0.1 0.1\n")  # otra clase
        escribir(lab(5), "0 0.5 0.5 0 0.1\n")  # ancho cero
        items, cuenta = A.collect(val)
        self.assertEqual(cuenta, {"con_caja": 6, "sin_raton": 1, "varias_cajas": 1, "invalidas": 3, "sin_etiqueta": 1})

    def test_dos_videos_y_formato_png(self):
        val, vt = self.etiquetado(claves=("A", "B"), por_video=5, formato="png")
        items, cuenta = A.collect(val)
        self.assertEqual(cuenta["con_caja"], 10)
        self.assertEqual(len({i.id for i in items}), 2)
        self.assertTrue(all(i.imagen.endswith(".png") and os.path.exists(i.imagen) for i in items))

    def test_carpeta_que_no_es_un_conjunto(self):
        with self.assertRaisesRegex(FileNotFoundError, "falta manifest.json"):
            A.collect(self.w)

    def correr(self, val, **kw):
        msgs = []
        rc = A.audit(val, out=msgs.append, **kw)
        return rc, "\n".join(msgs)

    def test_informe_y_hoja(self):
        val, _ = self.etiquetado(cajas=lambda f: (60, 70, 30, 20))  # todas iguales
        rc, texto = self.correr(val)
        self.assertEqual(rc, 0, msg=texto)
        self.assertIn("12 imágenes: 12 con caja, 0 sin ratón, 0 con varias cajas, 0 inválidas, 0 sin etiqueta", texto)
        self.assertIn("Ancho mediano 30 px, alto mediano 20 px", texto)
        self.assertIn("Área: mínima 600, mediana 600, máxima 600 px² (máx/mín = 1.0)", texto)
        self.assertIn("Ninguna caja sale de lo habitual", texto)
        hoja = cv2.imread(os.path.join(val, "auditoria.png"))
        self.assertEqual(hoja.shape, (2 * 200, 8 * 200, 3))  # 12 cajas en 8 columnas de 200 px
        self.assertIn("ventana de 45 px por celda", texto)  # ceil(30 * 1.5): el lado mayor de la caja mayor, por el contexto

    def test_una_caja_enorme_se_marca_con_su_nombre(self):
        def cajas(f):
            return (40, 40, 100, 100) if f == 30 else (60 + f % 5, 70, 30, 20)
        val, vt = self.etiquetado(cajas=cajas, por_video=40)
        assert 30 in vt[0]["frames"]
        rc, texto = self.correr(val)
        self.assertEqual(rc, 0)
        self.assertIn("Posibles inconsistentes", texto)
        self.assertIn("bg.avi f30 (100x100)", texto)
        self.assertNotIn("f32 (", texto)
        self.assertIn("(máx/mín = 16.7)", texto)  # 10000 / 600

    def test_pocas_cajas_avisa_que_no_marca(self):
        val, _ = self.etiquetado(por_video=5)
        rc, texto = self.correr(val)
        self.assertEqual(rc, 0)
        self.assertIn("menos de 8 cajas", texto)
        self.assertNotIn("Posibles inconsistentes", texto)

    def test_orden_columnas_celda_y_salida(self):
        val, _ = self.etiquetado()
        destino = os.path.join(self.w, "otra", "hoja.png")
        os.makedirs(os.path.dirname(destino))
        rc, texto = self.correr(val, orden="frame", columnas=4, celda=100, salida=destino)
        self.assertEqual(rc, 0)
        self.assertEqual(cv2.imread(destino).shape, (3 * 100, 4 * 100, 3))
        self.assertIn(destino, texto)
        self.assertIn("(frame,", texto)
        self.assertFalse(os.path.exists(os.path.join(val, "auditoria.png")))

    def test_errores(self):
        val, _ = self.etiquetado()
        for kw in ({"orden": "tamaño"}, {"columnas": 0}, {"celda": 10}, {"contexto": 0.5}):
            self.assertEqual(self.correr(val, **kw)[0], 2, msg=str(kw))
        rc, texto = self.correr(self.w)
        self.assertEqual(rc, 2)
        self.assertIn("falta manifest.json", texto)
        vacio, _ = self.etiquetado(etiquetar=False)
        rc, texto = self.correr(vacio)
        self.assertEqual(rc, 2)
        self.assertIn("No hay ninguna caja", texto)
        self.assertIn("sin etiqueta (dudosas o pendientes)", texto)
        self.assertEqual(self.correr(val, salida=os.path.join(self.w, "no_existe", "x.png"))[0], 2)  # no se puede escribir

    def test_cli(self):
        val, _ = self.etiquetado()
        self.assertEqual(A.main([val, "--orden", "aspecto", "--columnas", "6", "--celda", "120"]), 0)
        self.assertEqual(cv2.imread(os.path.join(val, "auditoria.png")).shape, (2 * 120, 6 * 120, 3))
        import contextlib, io
        for orden in ("frame", "aspecto", "area"):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(A.main([val, "--orden", orden]), 0)
            self.assertIn(f"Hoja ({orden},", buf.getvalue())  # el orden pedido en la línea de comandos llega hasta el final
        self.assertEqual(A.main([self.w]), 2)


if __name__ == "__main__":
    unittest.main()