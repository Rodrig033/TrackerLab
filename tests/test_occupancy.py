import random
import unittest

import numpy as np

from analysis import FrameRecord
from analysis.occupancy import OccupancyMap, average_maps, detected_points, occupancy_map


def rec(i, x, y):
    """Registro con el sujeto en (x, y) (coordenadas del CSV, Y hacia arriba); None = perdido."""
    if x is None:
        return FrameRecord(frame=i, time_s=i * 0.1, dist_acum_px=0.0)
    return FrameRecord(frame=i, time_s=i * 0.1, dist_acum_px=0.0, cx_abs=float(x), cy_abs_inv=float(y))


def occ(records, dt=0.1, sigma=0.0, size=100, bin_size=10):
    return occupancy_map(records, dt, (0, size), (0, size), bin_size, sigma)


class AsignacionTests(unittest.TestCase):
    def test_cada_punto_cae_en_su_celda(self):
        m = occ([rec(0, 5, 5), rec(1, 15, 5), rec(2, 15, 95), rec(3, 95, 95)])
        self.assertEqual(m.shape, (10, 10))
        s = m.seconds
        self.assertAlmostEqual(s[0, 0], 0.1)  # (x=5, y=5)   -> fila 0, columna 0
        self.assertAlmostEqual(s[0, 1], 0.1)  # (x=15, y=5)  -> fila 0, columna 1
        self.assertAlmostEqual(s[9, 1], 0.1)  # (x=15, y=95) -> fila 9 (arriba), columna 1
        self.assertAlmostEqual(s[9, 9], 0.1)
        self.assertAlmostEqual(float(s.sum()), 0.4)

    def test_el_tiempo_es_el_numero_de_frames_por_dt(self):
        pts = [rec(i, 33, 44) for i in range(25)]
        m = occ(pts, dt=0.04)
        self.assertAlmostEqual(m.total_time_s, 25 * 0.04)
        self.assertAlmostEqual(float(m.seconds.sum()), 1.0)
        self.assertAlmostEqual(float(m.seconds[4, 3]), 1.0)  # todo en una celda

    def test_los_frames_perdidos_no_aportan_tiempo(self):
        m = occ([rec(0, 5, 5), rec(1, None, None), rec(2, None, None), rec(3, 5, 5)])
        self.assertEqual(m.n_points, 2)
        self.assertAlmostEqual(m.total_time_s, 0.2)

    def test_el_borde_superior_se_incluye_y_lo_de_fuera_se_descarta(self):
        m = occ([rec(0, 100, 100), rec(1, 150, 50), rec(2, -1, 50)])
        self.assertEqual((m.n_points, m.n_outside), (3, 2))
        self.assertAlmostEqual(float(m.seconds[9, 9]), 0.1)  # x=100,y=100 cae en la última celda
        self.assertAlmostEqual(m.total_time_s, 0.1)

    def test_rangos_que_no_son_multiplos_del_bin(self):
        m = occupancy_map([rec(0, 25, 5)], 0.1, (0, 25), (0, 10), 10)
        self.assertEqual(m.shape, (1, 3))  # 25 / 10 -> 3 columnas (la última sobresale)
        self.assertAlmostEqual(float(m.seconds[0, 2]), 0.1)

    def test_campos_alternativos(self):
        r = FrameRecord(frame=0, time_s=0.0, dist_acum_px=0.0, cx_abs=500.0, cy_abs_inv=500.0, x_rel_cm=3.0, y_rel_cm=4.0)
        m = occupancy_map([r], 0.1, (0, 10), (0, 10), 1.0, x_field="x_rel_cm", y_field="y_rel_cm")
        self.assertAlmostEqual(float(m.seconds[4, 3]), 0.1)

    def test_detected_points_ignora_none_y_nan(self):
        r = [rec(0, 1, 2), rec(1, None, None), rec(2, float("nan"), 3), rec(3, 4, 5)]
        xs, ys = detected_points(r)
        self.assertEqual((list(xs), list(ys)), ([1.0, 4.0], [2.0, 5.0]))

    def test_validaciones(self):
        for kw in (dict(dt=0), dict(dt=-1)):
            with self.assertRaises(ValueError):
                occupancy_map([], kw["dt"], (0, 1), (0, 1), 1)
        with self.assertRaises(ValueError):
            occupancy_map([], 0.1, (0, 1), (0, 1), 0)
        with self.assertRaises(ValueError):
            occupancy_map([], 0.1, (5, 5), (0, 1), 1)
        with self.assertRaises(ValueError):
            occupancy_map([], 0.1, (0, 1), (0, 1), 1, sigma=-1)

    def test_sin_datos_da_un_mapa_vacio(self):
        m = occ([])
        self.assertEqual((m.n_points, m.total_time_s, float(m.seconds.sum())), (0, 0.0, 0.0))
        self.assertEqual(float(m.percent().sum()), 0.0)


class SuavizadoTests(unittest.TestCase):
    def test_el_suavizado_conserva_el_tiempo_total(self):
        rng = random.Random(0)
        pts = [rec(i, rng.uniform(0, 100), rng.uniform(0, 100)) for i in range(400)]
        for sigma in (5, 10, 20):
            m = occ(pts, sigma=sigma)
            self.assertAlmostEqual(float(m.seconds.sum()), m.total_time_s, places=9, msg=f"sigma={sigma}")

    def test_el_suavizado_conserva_el_tiempo_pegado_a_la_pared_y_a_la_esquina(self):
        """Donde un borde mal tratado perdería datos: animales que se pegan a la pared."""
        esquina = [rec(i, 1, 1) for i in range(50)]
        pared = [rec(i, 0.5, 50) for i in range(50)]
        for pts in (esquina, pared):
            m = occ(pts, sigma=15)
            self.assertAlmostEqual(float(m.seconds.sum()), 5.0, places=9)

    def test_el_suavizado_reparte_el_tiempo_y_mantiene_el_maximo_en_el_punto(self):
        m0 = occ([rec(i, 55, 55) for i in range(30)])
        m1 = occ([rec(i, 55, 55) for i in range(30)], sigma=10)
        self.assertEqual((m0.seconds > 0).sum(), 1)
        self.assertGreater((m1.seconds > 1e-6).sum(), 9)
        self.assertEqual(np.unravel_index(m1.seconds.argmax(), m1.shape), (5, 5))

    def test_el_suavizado_es_simetrico(self):
        m = occ([rec(i, 55, 55) for i in range(10)], sigma=10)
        s = m.seconds
        self.assertTrue(np.allclose(s, s.T))  # punto en la diagonal: simétrico en x e y
        self.assertTrue(np.allclose(s[4, 5], s[6, 5]))

    def test_sigma_grande_en_una_cuadricula_pequena_no_falla_y_conserva(self):
        m = occupancy_map([rec(0, 3, 3), rec(1, 8, 2)], 0.1, (0, 10), (0, 10), 5.0, sigma=50)  # 2x2 celdas
        self.assertAlmostEqual(float(m.seconds.sum()), 0.2, places=9)


class PorcentajeYGrupoTests(unittest.TestCase):
    def test_el_porcentaje_suma_100(self):
        m = occ([rec(0, 5, 5), rec(1, 55, 55), rec(2, 55, 55), rec(3, 95, 5)], sigma=7)
        self.assertAlmostEqual(float(m.percent().sum()), 100.0)

    def test_animales_grabados_distinto_tiempo_son_comparables_en_porcentaje(self):
        corto = occ([rec(i, 25, 25) for i in range(10)], dt=0.1)
        largo = occ([rec(i, 25, 25) for i in range(100)], dt=0.1)
        self.assertNotAlmostEqual(corto.total_time_s, largo.total_time_s)
        self.assertTrue(np.allclose(corto.percent(), largo.percent()))

    def test_promedio_de_grupo(self):
        a = occ([rec(i, 15, 15) for i in range(20)])
        b = occ([rec(i, 85, 85) for i in range(80)])
        prom = average_maps([a, b])
        self.assertAlmostEqual(float(prom.sum()), 100.0)
        self.assertAlmostEqual(float(prom[1, 1]), 50.0)  # cada animal pesa igual, sin importar el tiempo
        self.assertAlmostEqual(float(prom[8, 8]), 50.0)
        crudo = average_maps([a, b], normalize=False)
        self.assertAlmostEqual(float(crudo.sum()), (2.0 + 8.0) / 2)

    def test_promedio_exige_la_misma_cuadricula(self):
        a = occ([rec(0, 5, 5)])
        for otra in (occ([rec(0, 5, 5)], bin_size=20), occ([rec(0, 5, 5)], size=200)):
            with self.assertRaises(ValueError):
                average_maps([a, otra])
        with self.assertRaises(ValueError):
            average_maps([])


class PruebaConPipelineTests(unittest.TestCase):
    def test_el_tiempo_del_mapa_coincide_con_el_resumen_cinematico(self):
        """El tiempo total del mapa debe ser el 'tiempo_valido_s' que reporta el pipeline."""
        import importlib
        import os
        import tempfile

        import cv2

        from main import TrackingJob, run_tracking
        from tracking.background_tracker import DEFAULT_PARAMS, BackgroundTracker, compute_background
        from video.frame_selection import FrameRange, compute_timing, read_background_frames
        from video.reader import VideoReader

        try:
            BG = importlib.import_module("tests.test_background_tracker")
        except ImportError:
            BG = importlib.import_module("test_background_tracker")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "v.avi")
            frames, _ = BG.make_frames(60, lost={20, 21, 22})
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (BG.FW, BG.FH))
            for f in frames:
                vw.write(f)
            vw.release()
            mask = BG.roi_mask("elipse")
            with VideoReader(path) as r:
                bg = compute_background(read_background_frames(r, 30), BG.ROI, mask, False)
                job = TrackingJob(r, FrameRange(0, 55), compute_timing(r.fps), BackgroundTracker(mask, bg, DEFAULT_PARAMS),
                                  BG.ROI, start=((40.0, 90.0), (28, 78, 24, 24)))
                res = run_tracking(job, keep_records=True)
            m = occupancy_map(res.records, res.timing.dt, (0, BG.FW), (0, BG.FH), 8.0, sigma=12.0)
            self.assertEqual(m.n_outside, 0)
            self.assertAlmostEqual(m.total_time_s, res.kinematic.tiempo_valido_s, places=9)
            self.assertAlmostEqual(float(m.seconds.sum()), res.kinematic.tiempo_valido_s, places=9)


if __name__ == "__main__":
    unittest.main()