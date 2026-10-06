"""Pruebas de analysis/kinematic_metrics.py.

Ejecutar desde la raíz del proyecto:
    python -m pytest tests/test_kinematic_metrics.py -v
"""

import dataclasses
import random
import unittest

import numpy as np

from analysis import FrameRecord
from analysis.kinematic_metrics import (
    KinematicTracker,
    angle_between_vectors_deg,
    euclidean,
)

def _orig_euclidean(p1, p2):
    return float(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5)


def _orig_angle(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 <= 1e-12 or n2 <= 1e-12:
        return None
    cosang = np.dot(v1, v2) / (n1 * n2)
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    return float(ang)


def original_kinematics(points, dt, scale_info, reset_before=()):
    prev_center_abs = None
    prev_vel_px_s = None
    prev_vel_cm_s = None
    prev_point_for_ang = None
    dist_acum_px = 0.0
    dist_acum_cm = 0.0 if scale_info is not None else None
    valid_detected_frames = 0
    global_sum_vel_px = 0.0
    global_sum_vel_cm = 0.0
    global_n_vel = 0
    global_sum_acc_px = 0.0
    global_sum_acc_cm = 0.0
    global_n_acc = 0
    global_sum_ang = 0.0
    global_n_ang = 0
    global_sum_angvel = 0.0
    global_n_angvel = 0

    rows = []
    for i, pt in enumerate(points):
        if i in reset_before:  # tecla 'r'
            prev_center_abs = None
            prev_vel_px_s = None
            prev_vel_cm_s = None
            prev_point_for_ang = None

        found = pt is not None
        dist_px = None
        dist_cm = None
        vel_px_s = None
        vel_cm_s = None
        acel_px_s2 = None
        acel_cm_s2 = None
        ang_deg = None
        vel_angular_deg_s = None
        if found:
            valid_detected_frames += 1
            current_pt = pt
            if prev_center_abs is not None:
                dist_px = _orig_euclidean(current_pt, prev_center_abs)
                dist_acum_px += dist_px

                vel_px_s = dist_px / dt
                global_sum_vel_px += vel_px_s
                global_n_vel += 1

                if prev_vel_px_s is not None:
                    acel_px_s2 = (vel_px_s - prev_vel_px_s) / dt
                    global_sum_acc_px += acel_px_s2
                    global_n_acc += 1

                if scale_info is not None:
                    dist_cm = dist_px * scale_info["cm_per_px"]
                    dist_acum_cm += dist_cm

                    vel_cm_s = dist_cm / dt
                    global_sum_vel_cm += vel_cm_s

                    if prev_vel_cm_s is not None:
                        acel_cm_s2 = (vel_cm_s - prev_vel_cm_s) / dt
                        global_sum_acc_cm += acel_cm_s2
            else:
                dist_px = 0.0
                vel_px_s = None
                acel_px_s2 = None
                if scale_info is not None:
                    dist_cm = 0.0

            if prev_point_for_ang is not None and prev_center_abs is not None:
                v1 = np.array([prev_center_abs[0] - prev_point_for_ang[0],
                               prev_center_abs[1] - prev_point_for_ang[1]], dtype=float)
                v2 = np.array([current_pt[0] - prev_center_abs[0],
                               current_pt[1] - prev_center_abs[1]], dtype=float)
                ang_deg = _orig_angle(v1, v2)
                if ang_deg is not None:
                    vel_angular_deg_s = ang_deg / dt
                    global_sum_ang += ang_deg
                    global_n_ang += 1
                    global_sum_angvel += vel_angular_deg_s
                    global_n_angvel += 1

            prev_point_for_ang = prev_center_abs
            prev_center_abs = current_pt
            prev_vel_px_s = vel_px_s
            prev_vel_cm_s = vel_cm_s
        else:
            prev_center_abs = None
            prev_vel_px_s = None
            prev_vel_cm_s = None
            prev_point_for_ang = None

        rows.append((dist_acum_px, dist_acum_cm, dist_px, dist_cm, vel_px_s, vel_cm_s,
                     acel_px_s2, acel_cm_s2, ang_deg, vel_angular_deg_s))

    has_cm = scale_info is not None
    summary = (
        valid_detected_frames,
        valid_detected_frames * dt,
        dist_acum_px,
        dist_acum_cm,
        global_sum_vel_px / global_n_vel if global_n_vel > 0 else None,
        global_sum_vel_cm / global_n_vel if (has_cm and global_n_vel > 0) else None,
        global_sum_acc_px / global_n_acc if global_n_acc > 0 else None,
        global_sum_acc_cm / global_n_acc if (has_cm and global_n_acc > 0) else None,
        global_sum_ang / global_n_ang if global_n_ang > 0 else None,
        global_sum_angvel / global_n_angvel if global_n_angvel > 0 else None,
    )
    return rows, summary


def new_kinematics(points, dt, cm_per_px, reset_before=()):
    kin = KinematicTracker(dt=dt, cm_per_px=cm_per_px)
    rows = []
    for i, pt in enumerate(points):
        if i in reset_before:
            kin.reset_history()
        s = kin.update(pt)
        rows.append((s.dist_acum_px, s.dist_acum_cm, s.dist_px, s.dist_cm, s.vel_px_s,
                     s.vel_cm_s, s.acel_px_s2, s.acel_cm_s2, s.ang_deg, s.vel_angular_deg_s))
    m = kin.summary()
    summary = (m.frames_validos, m.tiempo_valido_s, m.dist_total_px, m.dist_total_cm,
               m.vel_media_px_s, m.vel_media_cm_s, m.acel_media_px_s2, m.acel_media_cm_s2,
               m.angulo_medio_deg, m.vel_angular_media_deg_s)
    return rows, summary


def random_track(rng, n):
    """Trayectoria con pérdidas, quietud, giros bruscos y tramos rectos."""
    pts, x, y = [], 300.0, 200.0
    dx, dy = rng.uniform(-3, 3), rng.uniform(-3, 3)
    for _ in range(n):
        r = rng.random()
        if r < 0.10:
            pts.append(None)  # sujeto perdido
            continue
        if r < 0.20:
            pass  # quieto: mismo punto que antes
        elif r < 0.30:
            dx, dy = rng.uniform(-15, 15), rng.uniform(-15, 15)  # giro brusco
            x, y = x + dx, y + dy
        else:
            x, y = x + dx, y + dy  # sigue recto (casi colineal)
        pts.append((x, y))
    return pts


class CoincideConElOriginalTests(unittest.TestCase):
    def test_trayectorias_aleatorias_con_y_sin_escala(self):
        for seed in range(60):
            rng = random.Random(seed)
            pts = random_track(rng, 250)
            resets = {i for i in range(250) if rng.random() < 0.02}
            dt = rng.choice([1 / 30, 1 / 25, 0.1, 3 / 29.97])
            for cm in (None, 0.0731, 1.5):
                scale = None if cm is None else {"cm_per_px": cm}
                exp = original_kinematics(pts, dt, scale, resets)
                got = new_kinematics(pts, dt, cm, resets)
                self.assertEqual(got[0], exp[0], msg=f"filas: seed={seed} cm={cm}")
                self.assertEqual(got[1], exp[1], msg=f"resumen: seed={seed} cm={cm}")

    def test_angulo_coincide_en_casos_limite(self):
        casos = [((1, 0), (1, 0)), ((1, 0), (-1, 0)), ((1, 0), (0, 1)),
                 ((0, 0), (1, 1)), ((1, 1), (0, 0)), ((1e-13, 0), (1, 0)),
                 ((3, 4), (3.0000001, 4.0000001))]
        for v1, v2 in casos:
            self.assertEqual(angle_between_vectors_deg(v1, v2),
                             _orig_angle(np.array(v1, float), np.array(v2, float)))


class ComportamientoTests(unittest.TestCase):
    def setUp(self):
        self.k = KinematicTracker(dt=0.5, cm_per_px=0.1)

    def test_primer_punto(self):
        s = self.k.update((10.0, 10.0))
        self.assertEqual((s.dist_px, s.dist_cm), (0.0, 0.0))
        self.assertIsNone(s.vel_px_s)
        self.assertIsNone(s.vel_cm_s)
        self.assertIsNone(s.acel_px_s2)
        self.assertIsNone(s.ang_deg)

    def test_velocidad_desde_el_segundo_aceleracion_desde_el_tercero(self):
        self.k.update((0.0, 0.0))
        s2 = self.k.update((3.0, 4.0))  # distancia 5
        self.assertEqual(s2.dist_px, 5.0)
        self.assertEqual(s2.vel_px_s, 10.0)
        self.assertAlmostEqual(s2.vel_cm_s, 1.0)
        self.assertIsNone(s2.acel_px_s2)
        self.assertIsNone(s2.ang_deg)  # el ángulo necesita 3 puntos
        s3 = self.k.update((3.0, 10.0))  # distancia 6, rapidez 12
        self.assertEqual(s3.vel_px_s, 12.0)
        self.assertEqual(s3.acel_px_s2, 4.0)  # (12 - 10) / 0.5
        self.assertIsNotNone(s3.ang_deg)

    def test_angulo_es_el_giro_entre_desplazamientos(self):
        self.k.update((0.0, 0.0))
        self.k.update((1.0, 0.0))
        s = self.k.update((1.0, 1.0))  # gira 90 grados
        self.assertAlmostEqual(s.ang_deg, 90.0)
        self.assertAlmostEqual(s.vel_angular_deg_s, 180.0)  # 90 / 0.5

    def test_quieto_no_tiene_angulo(self):
        self.k.update((0.0, 0.0))
        self.k.update((5.0, 0.0))
        s = self.k.update((5.0, 0.0))  # no se movió
        self.assertEqual(s.dist_px, 0.0)
        self.assertIsNone(s.ang_deg)

    def test_perdida_reinicia_memoria_pero_conserva_acumulados(self):
        self.k.update((0.0, 0.0))
        self.k.update((3.0, 4.0))
        lost = self.k.update(None)
        self.assertEqual(lost.dist_acum_px, 5.0)  # se conserva
        self.assertAlmostEqual(lost.dist_acum_cm, 0.5)
        for campo in ("dist_px", "dist_cm", "vel_px_s", "vel_cm_s", "acel_px_s2",
                      "acel_cm_s2", "ang_deg", "vel_angular_deg_s"):
            self.assertIsNone(getattr(lost, campo), campo)
        back = self.k.update((100.0, 100.0))  # reaparece lejos: no salta
        self.assertEqual(back.dist_px, 0.0)
        self.assertIsNone(back.vel_px_s)
        self.assertEqual(back.dist_acum_px, 5.0)  # la distancia no cruza el hueco

    def test_reset_history_conserva_acumulados(self):
        self.k.update((0.0, 0.0))
        self.k.update((3.0, 4.0))
        self.k.reset_history()
        s = self.k.update((50.0, 50.0))
        self.assertIsNone(s.vel_px_s)
        self.assertEqual(s.dist_acum_px, 5.0)

    def test_sin_escala_todo_en_cm_es_none(self):
        k = KinematicTracker(dt=0.5)
        k.update((0.0, 0.0))
        k.update((3.0, 4.0))
        s = k.update((6.0, 8.0))
        self.assertIsNone(s.dist_cm)
        self.assertIsNone(s.vel_cm_s)
        self.assertIsNone(s.acel_cm_s2)
        self.assertIsNone(s.dist_acum_cm)
        m = k.summary()
        self.assertIsNone(m.dist_total_cm)
        self.assertIsNone(m.vel_media_cm_s)
        self.assertIsNotNone(m.vel_media_px_s)

    def test_resumen_sin_datos(self):
        m = KinematicTracker(dt=0.5).summary()
        self.assertEqual((m.frames_validos, m.tiempo_valido_s, m.dist_total_px), (0, 0.0, 0.0))
        for campo in ("vel_media_px_s", "acel_media_px_s2", "angulo_medio_deg", "vel_angular_media_deg_s"):
            self.assertIsNone(getattr(m, campo), campo)

    def test_validaciones(self):
        for dt in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                KinematicTracker(dt=dt)
        with self.assertRaises(ValueError):
            KinematicTracker(dt=0.1, cm_per_px=0)

    def test_euclidean(self):
        self.assertEqual(euclidean((0, 0), (3, 4)), 5.0)


class ContratoConFrameRecordTests(unittest.TestCase):
    def test_sample_se_puede_volcar_en_frame_record(self):
        k = KinematicTracker(dt=0.1, cm_per_px=0.2)
        k.update((0.0, 0.0))
        s = k.update((3.0, 4.0))
        rec = FrameRecord(frame=1, time_s=0.1, cx_abs=3.0, cy_abs_inv=396.0, **dataclasses.asdict(s))
        self.assertTrue(rec.found)
        self.assertEqual(rec.vel_px_s, s.vel_px_s)

    def test_frame_perdido_tambien(self):
        k = KinematicTracker(dt=0.1)
        s = k.update(None)
        rec = FrameRecord(frame=1, time_s=0.1, **dataclasses.asdict(s))
        self.assertFalse(rec.found)


if __name__ == "__main__":
    unittest.main()