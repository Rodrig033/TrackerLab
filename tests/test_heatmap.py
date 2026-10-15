import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

try:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    from stats import heatmap as H
    from stats.heatmap import PlotContext, create_heatmap_plot, create_trajectory_plot, infer_cm_per_px

    HAVE_MPL = True
except ImportError:  # pragma: no cover
    HAVE_MPL = False

from analysis import FrameRecord
from analysis.occupancy import occupancy_map

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DT = 0.1


def rec(i, x, y, **kw):
    if x is None:
        return FrameRecord(frame=i, time_s=i * DT, dist_acum_px=0.0)
    return FrameRecord(frame=i, time_s=i * DT, dist_acum_px=0.0, cx_abs=float(x), cy_abs_inv=float(y), **kw)


def result_of(records):
    return SimpleNamespace(records=list(records), timing=SimpleNamespace(dt=DT),
                           kinematic=SimpleNamespace(dist_total_cm=None))


def walk(n=80, lost=()):
    """Recorrido que da la vuelta a la arena (400 x 300 px)."""
    out = []
    for i in range(n):
        if i in lost:
            out.append(rec(i, None, None))
            continue
        a = i / n * 2 * np.pi
        out.append(rec(i, 200 + 120 * np.cos(a), 150 + 90 * np.sin(a)))
    return out


def png_ok(path):
    with open(path, "rb") as f:
        sig = f.read(8)
    return sig == b"\x89PNG\r\n\x1a\n"


def rendered_array(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()


def colored_centroid(fig, margin=4):
    """Centroide (relativo 0..1, con Y hacia arriba) de los píxeles NO blancos dentro del área de ejes."""
    img = rendered_array(fig)
    ax = fig.axes[0]
    bb = ax.get_window_extent()
    h = img.shape[0]
    c0, c1 = int(bb.x0) + margin, int(bb.x1) - margin
    r0, r1 = int(h - bb.y1) + margin, int(h - bb.y0) - margin  # filas de la imagen (0 = arriba)
    zona = img[r0:r1, c0:c1]
    activo = (zona < 235).any(axis=2)
    rows, cols = np.nonzero(activo)
    assert len(rows) > 50, "no se dibujó nada dentro de los ejes"
    return cols.mean() / zona.shape[1], 1.0 - rows.mean() / zona.shape[0]


@unittest.skipUnless(HAVE_MPL, "requiere matplotlib")
class EscalaYVistaTests(unittest.TestCase):
    def test_infer_cm_per_px(self):
        regs = [rec(0, 1, 1, dist_px=10.0, dist_cm=0.7), rec(1, 2, 2, dist_px=20.0, dist_cm=1.4),
                rec(2, 3, 3, dist_px=0.0, dist_cm=0.0), rec(3, 4, 4)]
        self.assertAlmostEqual(infer_cm_per_px(regs), 0.07)
        self.assertIsNone(infer_cm_per_px([rec(0, 1, 1, dist_px=5.0)]))
        self.assertIsNone(infer_cm_per_px([]))

    def test_vista_en_px_cm_y_relativa(self):
        regs = [rec(0, 1, 1)]
        v = H._make_view(PlotContext(), regs)
        self.assertEqual((v.units, v.relative, v.x(10), v.y(20)), ("px", False, 10.0, 20.0))
        v = H._make_view(PlotContext(cm_per_px=0.5), regs)
        self.assertEqual((v.units, v.x(10), v.y(20)), ("cm", 5.0, 10.0))
        # origen en (100, 200) de la IMAGEN (Y hacia abajo) en un video de 300 px de alto: en Y hacia arriba es y=100
        v = H._make_view(PlotContext(cm_per_px=0.5, xy_origin=(100.0, 200.0), frame_size=(400, 300)), regs)
        self.assertTrue(v.relative)
        self.assertEqual((v.x(100), v.y(100)), (0.0, 0.0))  # el origen cae en (0, 0)
        self.assertEqual((v.x(120), v.y(130)), (10.0, 15.0))  # 20 px a la derecha, 30 px arriba

    def test_la_vista_relativa_coincide_con_las_columnas_x_rel_del_csv(self):
        """Lo que se grafica en modo relativo debe ser lo mismo que x_rel_cm / y_rel_cm."""
        H_VIDEO, ox, oy, cm = 300, 100.0, 200.0, 0.5
        cx, cy_img = 160.0, 120.0
        record = rec(0, cx, H_VIDEO - cy_img)
        x_rel_cm, y_rel_cm = (cx - ox) * cm, (oy - cy_img) * cm  # fórmula de main.py
        v = H._make_view(PlotContext(cm_per_px=cm, xy_origin=(ox, oy), frame_size=(400, H_VIDEO)), [record])
        self.assertAlmostEqual(v.x(record.cx_abs), x_rel_cm)
        self.assertAlmostEqual(v.y(record.cy_abs_inv), y_rel_cm)

    def test_cuadricula_del_roi_o_de_los_datos(self):
        xs, ys = np.array([10.0, 50.0]), np.array([20.0, 60.0])
        ctx = PlotContext(roi_box=(100, 40, 200, 150), frame_size=(400, 300))
        self.assertEqual(H._extent_px(ctx, xs, ys), ((100.0, 300.0), (110.0, 260.0)))  # y: 300-(40+150) .. 300-40
        (a, b), (c, d) = H._extent_px(PlotContext(), xs, ys)
        self.assertTrue(a < 10 and b > 50 and c < 20 and d > 60)  # datos con margen
        # ROI sin tamaño de frame: no se puede pasar a Y hacia arriba -> usa los datos
        self.assertEqual(H._extent_px(PlotContext(roi_box=(1, 1, 2, 2)), xs, ys), H._extent_px(PlotContext(), xs, ys))


@unittest.skipUnless(HAVE_MPL, "requiere matplotlib")
class SegmentosTests(unittest.TestCase):
    def test_un_hueco_corta_la_linea(self):
        v = H._make_view(PlotContext(), [])
        regs = [rec(0, 0, 0), rec(1, 1, 0), rec(2, None, None), rec(3, 5, 0), rec(4, 6, 0)]
        segs, times = H._segments(regs, v)
        self.assertEqual(len(segs), 2)  # 0-1 y 3-4; NO 1-3
        self.assertEqual(segs[1], [(5.0, 0.0), (6.0, 0.0)])
        self.assertEqual(times, [rec(1, 0, 0).time_s, rec(4, 0, 0).time_s])

    def test_sin_perdidas_hay_n_menos_uno_segmentos(self):
        self.assertEqual(len(H._segments(walk(50), H._make_view(PlotContext(), []))[0]), 49)


@unittest.skipUnless(HAVE_MPL, "requiere matplotlib")
class OrientacionTests(unittest.TestCase):
    """El error más fácil: invertir arriba y abajo. Se mide dónde cae el punto caliente."""

    CTX = PlotContext(roi_box=(0, 0, 400, 300), frame_size=(400, 300))

    def hotspot(self, x, y_up):
        regs = [rec(i, x, y_up) for i in range(40)]
        fig = H._render_heatmap(regs, DT, self.CTX, 1.0, 1.5, "t")
        return colored_centroid(fig)

    def test_cada_esquina_cae_en_su_cuadrante(self):
        casos = {
            "arriba-izquierda": ((60, 240), (True, True)),  # x pequeña, y_up grande
            "arriba-derecha": ((340, 240), (False, True)),
            "abajo-izquierda": ((60, 60), (True, False)),
            "abajo-derecha": ((340, 60), (False, False)),
        }
        for nombre, ((x, y), (izq, arriba)) in casos.items():
            cx, cy = self.hotspot(x, y)
            self.assertEqual(cx < 0.5, izq, msg=f"{nombre}: columna {cx:.2f}")
            self.assertEqual(cy > 0.5, arriba, msg=f"{nombre}: altura {cy:.2f}")

    def test_la_posicion_relativa_es_la_correcta(self):
        cx, cy = self.hotspot(100, 225)  # 25 % del ancho, 75 % del alto (Y hacia arriba)
        self.assertAlmostEqual(cx, 0.25, delta=0.08)
        self.assertAlmostEqual(cy, 0.75, delta=0.08)

    def test_con_el_fondo_del_video_la_orientacion_no_cambia(self):
        """El frame (con su fila 0 arriba) y el mapa deben quedar alineados: marcador a la izquierda-arriba del VIDEO."""
        fondo = np.zeros((300, 400, 3), np.uint8)
        cv2.circle(fondo, (60, 60), 25, (0, 255, 0), -1)  # en la imagen: arriba a la izquierda (y_img = 60)
        ctx = PlotContext(roi_box=(0, 0, 400, 300), frame_size=(400, 300), background_bgr=fondo)
        # sujeto exactamente sobre ese círculo: x=60, y_up = 300 - 60 = 240
        fig = H._render_heatmap([rec(i, 60, 240) for i in range(40)], DT, ctx, 1.0, 1.5, "t")
        img = rendered_array(fig)
        bb = fig.axes[0].get_window_extent()
        hh = img.shape[0]
        c0, c1, r0, r1 = int(bb.x0), int(bb.x1), int(hh - bb.y1), int(hh - bb.y0)
        verde = (img[r0:r1, c0:c1, 1] > 200) & (img[r0:r1, c0:c1, 0] < 80) & (img[r0:r1, c0:c1, 2] < 80)
        rows, cols = np.nonzero(verde)
        self.assertGreater(len(rows), 30)
        self.assertLess(cols.mean() / (c1 - c0), 0.5)
        self.assertLess(rows.mean() / (r1 - r0), 0.5)  # el círculo verde aparece ARRIBA a la izquierda


@unittest.skipUnless(HAVE_MPL, "requiere matplotlib")
class ZonasTests(unittest.TestCase):
    """Las zonas vienen como máscaras de IMAGEN (Y hacia abajo) y deben dibujarse con Y hacia arriba."""

    def zona(self, rect, alto=300, ancho=400):
        m = np.zeros((alto, ancho), np.uint8)
        cv2.rectangle(m, rect[:2], rect[2:], 255, -1)
        return SimpleNamespace(name="centro", mask=m)

    def test_el_contorno_y_la_etiqueta_usan_y_hacia_arriba(self):
        # rectángulo en la imagen: x 50..180, y 50..160 (cerca del borde SUPERIOR); video de 300 px de alto
        ctx = PlotContext(zones=[self.zona((50, 50, 180, 160))])
        fig, ax = H._new_figure()
        H._draw_zones(ax, ctx, H._make_view(ctx, []))
        ys = np.concatenate([np.asarray(l.get_ydata()) for l in ax.lines])
        xs = np.concatenate([np.asarray(l.get_xdata()) for l in ax.lines])
        self.assertEqual((ys.min(), ys.max()), (300 - 160, 300 - 50))  # 140 .. 250: la zona queda ARRIBA
        self.assertEqual((xs.min(), xs.max()), (50.0, 180.0))
        etiqueta = ax.texts[0]
        self.assertEqual(etiqueta.get_text(), "centro")
        self.assertAlmostEqual(etiqueta.get_position()[0], 115.0, delta=1.0)
        self.assertAlmostEqual(etiqueta.get_position()[1], 300 - 105.0, delta=1.0)  # centro en y_img=105 -> 195

    def test_la_zona_coincide_con_el_sujeto_que_esta_dentro_de_ella(self):
        """Un sujeto parado dentro de la zona (coordenadas del CSV) debe quedar dentro del contorno dibujado."""
        zona = self.zona((50, 50, 180, 160))
        sujeto_y_img = 100.0  # dentro de y 50..160
        record = rec(0, 100.0, 300 - sujeto_y_img)  # cy_abs_inv = alto - y_img
        ctx = PlotContext(zones=[zona])
        fig, ax = H._new_figure()
        v = H._make_view(ctx, [record])
        H._draw_zones(ax, ctx, v)
        ys = np.concatenate([np.asarray(l.get_ydata()) for l in ax.lines])
        self.assertTrue(ys.min() < v.y(record.cy_abs_inv) < ys.max())

    def test_se_aplica_la_escala_a_las_zonas(self):
        ctx = PlotContext(cm_per_px=0.5, zones=[self.zona((50, 50, 180, 160))])
        fig, ax = H._new_figure()
        H._draw_zones(ax, ctx, H._make_view(ctx, []))
        xs = np.concatenate([np.asarray(l.get_xdata()) for l in ax.lines])
        self.assertEqual((xs.min(), xs.max()), (25.0, 90.0))


@unittest.skipUnless(HAVE_MPL, "requiere matplotlib")
class FigurasTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_genera_png_validos_solo_con_result(self):
        res = result_of(walk(120, lost=set(range(30, 40))))
        for fn, nombre in ((create_heatmap_plot, "eda_mapa_calor_x.png"), (create_trajectory_plot, "eda_trayectoria_x.png")):
            path = fn(res, self.tmp.name, "x")
            self.assertEqual(os.path.basename(path), nombre)
            self.assertTrue(png_ok(path))
            img = cv2.imread(path)
            self.assertGreater(img.shape[0], 500)
            self.assertGreater(float(img.std()), 5.0)  # no es una imagen vacía

    def test_crea_la_carpeta_de_salida(self):
        destino = os.path.join(self.tmp.name, "a", "b")
        self.assertTrue(png_ok(create_heatmap_plot(result_of(walk()), destino, "x")))

    def test_sin_datos_suficientes_devuelve_none(self):
        for regs in ([], [rec(0, 1, 1)], [rec(i, None, None) for i in range(10)]):
            self.assertIsNone(create_heatmap_plot(result_of(regs), self.tmp.name, "x"))
            self.assertIsNone(create_trajectory_plot(result_of(regs), self.tmp.name, "x"))
        self.assertEqual(os.listdir(self.tmp.name), [])

    def test_unidades_en_las_etiquetas(self):
        px = H._render_heatmap(walk(), DT, PlotContext(), 1.0, 1.5)
        self.assertIn("(px)", px.axes[0].get_xlabel())
        con_escala = [rec(i, 200 + i, 150, dist_px=1.0, dist_cm=0.07) for i in range(50)]  # el CSV trae cm
        cm = H._render_heatmap(con_escala, DT, PlotContext(), 1.0, 1.5)
        self.assertIn("(cm)", cm.axes[0].get_xlabel())
        rel = H._render_heatmap(walk(), DT, PlotContext(cm_per_px=0.1, xy_origin=(10.0, 290.0), frame_size=(400, 300)), 1.0, 1.5)
        self.assertIn("relativo al origen", rel.axes[0].get_xlabel())

    def test_contexto_completo(self):
        fondo = np.full((300, 400, 3), 90, np.uint8)
        zona = np.zeros((300, 400), np.uint8)
        cv2.rectangle(zona, (50, 50), (180, 160), 255, -1)
        z = SimpleNamespace(name="centro", mask=zona)
        ctx = PlotContext(cm_per_px=0.07, frame_size=(400, 300), roi_box=(20, 20, 360, 260), xy_origin=(20.0, 280.0),
                          background_bgr=fondo, zones=[z])
        res = result_of(walk(100, lost={10, 11}))
        for fn in (create_heatmap_plot, create_trajectory_plot):
            self.assertTrue(png_ok(fn(res, self.tmp.name, "completo", context=ctx)))

    def test_la_nota_informa_los_puntos_fuera_de_la_cuadricula(self):
        ctx = PlotContext(roi_box=(0, 0, 400, 300), frame_size=(400, 300))
        regs = [rec(i, 100, 100) for i in range(20)] + [rec(100 + i, 900, 100) for i in range(5)]
        fig = H._render_heatmap(regs, DT, ctx, 1.0, 1.5)
        textos = " ".join(t.get_text() for t in fig.texts)
        self.assertIn("5 puntos fuera", textos)
        self.assertIn("2.0 s", textos)  # solo los 20 puntos dentro: 20 * 0.1

    def test_el_tiempo_graficado_es_el_del_mapa_de_ocupacion(self):
        regs = walk(100, lost={5, 6, 7})
        ctx = PlotContext(roi_box=(0, 0, 400, 300), frame_size=(400, 300))
        m = occupancy_map(regs, DT, (0, 400), (0, 300), 5.0)
        self.assertAlmostEqual(m.total_time_s, 9.7)
        textos = " ".join(t.get_text() for t in H._render_heatmap(regs, DT, ctx, 1.0, 1.5).texts)
        self.assertIn("9.7 s", textos)


@unittest.skipUnless(HAVE_MPL, "requiere matplotlib")
class SinEfectosGlobalesTests(unittest.TestCase):
    def test_no_importa_pyplot_ni_cambia_el_backend(self):
        codigo = ("import sys; import stats.heatmap; "
                  "print('matplotlib.pyplot' in sys.modules)")
        out = subprocess.run([sys.executable, "-c", codigo], cwd=ROOT, capture_output=True, text=True,
                             env={**os.environ, "PYTHONPATH": ROOT})
        self.assertEqual(out.returncode, 0, msg=out.stderr)
        self.assertEqual(out.stdout.strip(), "False")


@unittest.skipUnless(HAVE_MPL, "requiere matplotlib")
class IntegracionEdaTests(unittest.TestCase):
    def test_generate_eda_incluye_mapa_de_calor_y_trayectoria(self):
        try:
            from stats.eda import generate_eda
        except ImportError:
            self.skipTest("stats/eda.py (y descriptive.py, plots.py) no están en esta copia")
        regs = []
        for i in range(120):
            a = i / 120 * 2 * np.pi
            regs.append(FrameRecord(frame=i, time_s=i * DT, dist_acum_px=float(i), cx_abs=200 + 120 * np.cos(a),
                                    cy_abs_inv=150 + 90 * np.sin(a), dist_px=3.0, dist_cm=0.21, vel_px_s=30.0,
                                    vel_cm_s=2.1, acel_px_s2=1.0 + (i % 7), acel_cm_s2=0.07 * (1 + i % 7),
                                    ang_deg=5.0 + i % 11, vel_angular_deg_s=10.0 + i % 13))
        res = SimpleNamespace(records=regs, timing=SimpleNamespace(dt=DT), kinematic=SimpleNamespace(dist_total_cm=12.3))
        with tempfile.TemporaryDirectory() as tmp:
            out = generate_eda(res, tmp, "e", context=PlotContext(cm_per_px=0.07))
            self.assertEqual(set(out), {"summary", "velocity", "acceleration", "correlation", "heatmap", "trajectory"})
            for clave, ruta in out.items():
                self.assertTrue(ruta and os.path.exists(ruta), msg=clave)
            # sin contexto (como lo llamaban antes) sigue funcionando
            self.assertEqual(set(generate_eda(res, tmp, "e2")), set(out))


if __name__ == "__main__":
    unittest.main()