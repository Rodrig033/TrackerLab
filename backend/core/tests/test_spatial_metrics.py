import random
import unittest

import cv2
import numpy as np

from analysis import ZoneRecord
from analysis.spatial_metrics import ZoneGeometry, ZoneTracker, point_in_mask

W, H = 640, 480

def _o_euclidean(p1, p2):
    return float(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5)


def _o_point_in_mask(mask, x, y):
    h, w = mask.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if xi < 0 or xi >= w or yi < 0 or yi >= h:
        return 0
    return 1 if mask[yi, xi] > 0 else 0


def _o_distance_point_to_mask_perimeter(pt, mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    x, y = float(pt[0]), float(pt[1])
    inside_any = False
    min_abs_dist = None
    for c in cnts:
        d = cv2.pointPolygonTest(c, (x, y), True)
        if d >= 0:
            inside_any = True
        ad = abs(float(d))
        if min_abs_dist is None or ad < min_abs_dist:
            min_abs_dist = ad
    if inside_any:
        return 0.0
    return min_abs_dist


def original_zones(points, zones, dt, scale_info, start_idx, step, fps, reset_before=()):
    """points: lista de (x, y) o None, uno por frame procesado."""
    zone_stats = {}
    for z in zones:
        zone_stats[z["name"]] = {
            "time_s": 0.0, "frames_in_zone": 0, "entries": 0, "lat_first_s": None,
            "frame_first": None, "prev_inside": 0, "sum_dist_center_px": 0.0,
            "sum_dist_center_cm": 0.0, "sum_dist_perim_px": 0.0,
            "sum_dist_perim_cm": 0.0, "n_dist_valid": 0,
        }
    valid_detected_frames = 0
    rows = []
    for i, current_pt in enumerate(points):
        frame_idx = start_idx + i * step
        time_s = (frame_idx - start_idx) / float(fps)
        if i in reset_before:
            for z in zones:
                zone_stats[z["name"]]["prev_inside"] = None
        zone_values = []
        if current_pt is not None:
            valid_detected_frames += 1
            for z in zones:
                zn = z["name"]
                inside = _o_point_in_mask(z["mask_full"], current_pt[0], current_pt[1])
                d_center_px = None
                d_center_cm = None
                d_perim_px = None
                d_perim_cm = None
                if z["centroid_full"] is not None:
                    d_center_px = _o_euclidean(current_pt, z["centroid_full"])
                    if scale_info is not None:
                        d_center_cm = d_center_px * scale_info["cm_per_px"]
                d_perim_px = _o_distance_point_to_mask_perimeter(current_pt, z["mask_full"])
                if d_perim_px is not None and scale_info is not None:
                    d_perim_cm = d_perim_px * scale_info["cm_per_px"]
                if inside == 1:
                    zone_stats[zn]["time_s"] += dt
                    zone_stats[zn]["frames_in_zone"] += 1
                if zone_stats[zn]["prev_inside"] == 0 and inside == 1:
                    zone_stats[zn]["entries"] += 1
                    if zone_stats[zn]["lat_first_s"] is None:
                        zone_stats[zn]["lat_first_s"] = time_s
                        zone_stats[zn]["frame_first"] = frame_idx
                zone_stats[zn]["prev_inside"] = inside
                if d_center_px is not None:
                    zone_stats[zn]["sum_dist_center_px"] += d_center_px
                if d_center_cm is not None:
                    zone_stats[zn]["sum_dist_center_cm"] += d_center_cm
                if d_perim_px is not None:
                    zone_stats[zn]["sum_dist_perim_px"] += d_perim_px
                if d_perim_cm is not None:
                    zone_stats[zn]["sum_dist_perim_cm"] += d_perim_cm
                zone_stats[zn]["n_dist_valid"] += 1
                zone_values.append((
                    zn, inside, d_center_px, d_center_cm, d_perim_px, d_perim_cm,
                    zone_stats[zn]["time_s"], zone_stats[zn]["frames_in_zone"],
                    zone_stats[zn]["entries"], zone_stats[zn]["lat_first_s"],
                    zone_stats[zn]["frame_first"]))
        else:
            for z in zones:
                zone_stats[z["name"]]["prev_inside"] = None
        rows.append(tuple(zone_values))

    total_time_valid_s = valid_detected_frames * dt
    summary = []
    for z in zones:
        zn = z["name"]
        zs = zone_stats[zn]
        nvalid = zs["n_dist_valid"]
        summary.append((
            zn, zs["time_s"], zs["frames_in_zone"],
            zs["time_s"] / total_time_valid_s if total_time_valid_s > 0 else None,
            zs["entries"], zs["lat_first_s"], zs["frame_first"],
            zs["sum_dist_center_px"] / nvalid if nvalid > 0 else None,
            zs["sum_dist_center_cm"] / nvalid if (scale_info is not None and nvalid > 0) else None,
            zs["sum_dist_perim_px"] / nvalid if nvalid > 0 else None,
            zs["sum_dist_perim_cm"] / nvalid if (scale_info is not None and nvalid > 0) else None,
        ))
    return rows, tuple(summary)


def new_zones(points, zones, dt, cm_per_px, start_idx, step, fps, reset_before=()):
    geoms = [ZoneGeometry(z["name"], z["mask_full"], z["centroid_full"]) for z in zones]
    zt = ZoneTracker(geoms, dt=dt, cm_per_px=cm_per_px)
    rows = []
    for i, pt in enumerate(points):
        frame_idx = start_idx + i * step
        time_s = (frame_idx - start_idx) / float(fps)
        if i in reset_before:
            zt.reset_history()
        recs = zt.update(pt, frame_idx, time_s)
        rows.append(tuple(
            (r.name, r.inside, r.dist_center_px, r.dist_center_cm, r.dist_perim_px, r.dist_perim_cm,
             r.time_acum_s, r.frames_acum, r.entries_acum, r.lat_first_s, r.frame_first) for r in recs))
    summary = tuple(
        (s.name, s.tiempo_total_s, s.frames_total, s.proporcion_tiempo, s.entradas,
         s.lat_primera_entrada_s, s.frame_primera_entrada, s.dist_media_centro_px,
         s.dist_media_centro_cm, s.dist_media_perimetro_px, s.dist_media_perimetro_cm)
        for s in zt.summary())
    return rows, summary

def centroid_from_mask(mask):
    M = cv2.moments(mask)
    if M["m00"] <= 1e-12:
        return None
    return (float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"]))


def make_zone(name, drawer):
    m = np.zeros((H, W), np.uint8)
    drawer(m)
    return {"name": name, "mask_full": m, "centroid_full": centroid_from_mask(m)}


def sample_zones():
    return [
        make_zone("rect", lambda m: cv2.rectangle(m, (50, 50), (250, 200), 255, -1)),
        make_zone("circulo", lambda m: cv2.circle(m, (450, 300), 70, 255, -1)),
        make_zone("dona", lambda m: (cv2.circle(m, (150, 350), 80, 255, -1), cv2.circle(m, (150, 350), 35, 0, -1))),
        make_zone("islas", lambda m: (cv2.rectangle(m, (350, 40), (420, 100), 255, -1),
                                      cv2.rectangle(m, (520, 60), (600, 140), 255, -1))),
        make_zone("ele", lambda m: cv2.fillPoly(m, [np.array([[260, 220], [330, 220], [330, 400], [400, 400], [400, 440], [260, 440]])], 255)),
        make_zone("vacia", lambda m: None),
    ]


def random_points(rng, n):
    pts, x, y = [], 320.0, 240.0
    for _ in range(n):
        r = rng.random()
        if r < 0.08:
            pts.append(None)
            continue
        if r < 0.12:  # salto a una posición cualquiera, incluso fuera de la imagen
            x, y = rng.uniform(-30, W + 30), rng.uniform(-30, H + 30)
        else:
            x += rng.uniform(-25, 25)
            y += rng.uniform(-25, 25)
        if rng.random() < 0.2:  # coordenadas .5 para ejercitar el redondeo al par
            x, y = int(x) + 0.5, int(y) + 0.5
        pts.append((x, y))
    return pts

class CoincideConElOriginalTests(unittest.TestCase):
    def test_trayectorias_aleatorias_con_y_sin_escala(self):
        zones = sample_zones()
        for seed in range(30):
            rng = random.Random(seed)
            pts = random_points(rng, 220)
            resets = {i for i in range(220) if rng.random() < 0.02}
            fps = rng.choice([25.0, 29.97, 30.0])
            step = rng.choice([1, 2, 3])
            dt = step / fps
            start = rng.choice([0, 10, 137])
            for cm in (None, 0.0731):
                scale = None if cm is None else {"cm_per_px": cm}
                exp = original_zones(pts, zones, dt, scale, start, step, fps, resets)
                got = new_zones(pts, zones, dt, cm, start, step, fps, resets)
                self.assertEqual(got[0], exp[0], msg=f"filas: seed={seed} cm={cm}")
                self.assertEqual(got[1], exp[1], msg=f"resumen: seed={seed} cm={cm}")

    def test_la_prueba_ejercita_los_casos_dificiles(self):
        """Evita que la comparación pase 'en vacío'."""
        zones = sample_zones()
        total = dentro = entradas = 0
        for seed in range(30):
            pts = random_points(random.Random(seed), 220)
            rows, summ = new_zones(pts, zones, 0.04, 0.07, 0, 1, 25.0)
            for fila in rows:
                for z in fila:
                    total += 1
                    dentro += z[1]
            entradas += sum(s[4] for s in summ)
        self.assertGreater(dentro, 500)
        self.assertGreater(entradas, 100)


class ZoneTrackerTests(unittest.TestCase):
    def setUp(self):
        m = np.zeros((100, 100), np.uint8)
        cv2.rectangle(m, (40, 40), (60, 60), 255, -1)
        self.geom = ZoneGeometry("z", m, (50.0, 50.0))

    def tracker(self, **kw):
        return ZoneTracker([self.geom], dt=kw.pop("dt", 0.5), cm_per_px=kw.pop("cm_per_px", None))

    def test_si_arranca_dentro_cuenta_como_entrada(self):
        zt = self.tracker()
        r = zt.update((50, 50), frame_idx=100, time_s=0.0)[0]
        self.assertEqual((r.inside, r.entries_acum, r.lat_first_s, r.frame_first), (1, 1, 0.0, 100))

    def test_entrada_latencia_y_acumulados(self):
        zt = self.tracker()
        zt.update((10, 10), 0, 0.0)
        zt.update((10, 10), 1, 0.5)
        r = zt.update((50, 50), 2, 1.0)[0]
        self.assertEqual((r.entries_acum, r.lat_first_s, r.frame_first), (1, 1.0, 2))
        r = zt.update((51, 51), 3, 1.5)[0]
        self.assertEqual((r.frames_acum, r.time_acum_s, r.entries_acum), (2, 1.0, 1))
        zt.update((10, 10), 4, 2.0)
        r = zt.update((50, 50), 5, 2.5)[0]
        self.assertEqual((r.entries_acum, r.lat_first_s), (2, 1.0))  # la latencia no cambia

    def test_perdida_no_cuenta_entrada_falsa_al_reaparecer_dentro(self):
        zt = self.tracker()
        zt.update((10, 10), 0, 0.0)
        zt.update((50, 50), 1, 0.5)  # entrada 1
        self.assertEqual(zt.update(None, 2, 1.0), ())
        r = zt.update((50, 50), 3, 1.5)[0]  # reaparece dentro
        self.assertEqual(r.entries_acum, 1)

    def test_perdida_y_luego_entrada_real_si_cuenta(self):
        zt = self.tracker()
        zt.update((10, 10), 0, 0.0)
        zt.update(None, 1, 0.5)
        zt.update((10, 10), 2, 1.0)  # reaparece fuera: estado conocido = 0
        r = zt.update((50, 50), 3, 1.5)[0]
        self.assertEqual(r.entries_acum, 1)

    def test_reset_history_conserva_acumulados(self):
        zt = self.tracker()
        zt.update((50, 50), 0, 0.0)  # entrada 1
        zt.reset_history()
        r = zt.update((50, 50), 1, 0.5)[0]
        self.assertEqual((r.entries_acum, r.frames_acum), (1, 2))

    def test_distancias_y_escala(self):
        zt = self.tracker(cm_per_px=0.1)
        r = zt.update((50, 80), 0, 0.0)[0]
        self.assertEqual(r.inside, 0)
        self.assertEqual(r.dist_center_px, 30.0)
        self.assertAlmostEqual(r.dist_center_cm, 3.0)
        self.assertEqual(r.dist_perim_px, 20.0)  # el borde está en y=60
        self.assertAlmostEqual(r.dist_perim_cm, 2.0)

    def test_sin_escala_cm_es_none(self):
        r = self.tracker().update((50, 80), 0, 0.0)[0]
        self.assertIsNone(r.dist_center_cm)
        self.assertIsNone(r.dist_perim_cm)
        s = self.tracker().summary()[0]
        self.assertIsNone(s.dist_media_centro_cm)

    def test_dentro_tiene_distancia_al_perimetro_cero(self):
        self.assertEqual(self.tracker().update((50, 50), 0, 0.0)[0].dist_perim_px, 0.0)

    def test_resumen(self):
        zt = self.tracker()
        zt.update((50, 50), 0, 0.0)  # dentro
        zt.update(None, 1, 0.5)  # perdido: no cuenta como válido
        zt.update((10, 10), 2, 1.0)  # fuera
        s = zt.summary()[0]
        self.assertEqual(s.tiempo_total_s, 0.5)
        self.assertAlmostEqual(s.proporcion_tiempo, 0.5)  # 0.5 s de 1.0 s válido
        self.assertEqual((s.entradas, s.frames_total), (1, 1))

    def test_sin_zonas(self):
        zt = ZoneTracker([], dt=0.1)
        self.assertEqual(zt.update((1, 1), 0, 0.0), ())
        self.assertEqual(zt.summary(), ())

    def test_registros_son_zone_record(self):
        r = self.tracker().update((50, 50), 0, 0.0)
        self.assertIsInstance(r[0], ZoneRecord)
        self.assertEqual(r[0].name, "z")

    def test_validaciones(self):
        with self.assertRaises(ValueError):
            ZoneTracker([self.geom, self.geom], dt=0.1)  # nombre repetido
        with self.assertRaises(ValueError):
            ZoneTracker([self.geom], dt=0)
        with self.assertRaises(ValueError):
            ZoneTracker([self.geom], dt=0.1, cm_per_px=-1)
        with self.assertRaises(ValueError):
            ZoneGeometry("x", np.zeros((3, 4, 5), np.uint8))


class ComportamientosDelOriginalTests(unittest.TestCase):
    """Rarezas del original que se conservan a propósito (candidatas a revisar)."""

    def test_punto_en_el_hueco_esta_fuera_pero_perimetro_cero(self):
        m = np.zeros((200, 200), np.uint8)
        cv2.circle(m, (100, 100), 80, 255, -1)
        cv2.circle(m, (100, 100), 30, 0, -1)
        g = ZoneGeometry("dona", m, (100.0, 100.0))
        self.assertEqual(g.contains((100, 100)), 0)  # en el hueco: fuera
        self.assertEqual(g.perimeter_distance((100, 100)), 0.0)  # pero 'dentro' del contorno externo

    def test_zona_vacia_promedia_cero_en_vez_de_vacio(self):
        g = ZoneGeometry("v", np.zeros((50, 50), np.uint8), None)
        zt = ZoneTracker([g], dt=0.1)
        r = zt.update((10, 10), 0, 0.0)[0]
        self.assertIsNone(r.dist_center_px)
        self.assertIsNone(r.dist_perim_px)
        s = zt.summary()[0]
        self.assertEqual((s.dist_media_centro_px, s.dist_media_perimetro_px), (0.0, 0.0))

    def test_redondeo_al_par(self):
        m = np.zeros((10, 10), np.uint8)
        m[5, 2] = 255
        self.assertEqual(point_in_mask(m, 2.5, 5), 1)  # round(2.5) == 2
        self.assertEqual(point_in_mask(m, 3.5, 5), 0)  # round(3.5) == 4

    def test_fuera_de_la_imagen(self):
        m = np.full((10, 10), 255, np.uint8)
        for p in ((-1, 5), (5, -1), (10, 5), (5, 10), (-0.6, 5)):
            self.assertEqual(point_in_mask(m, *p), 0, msg=str(p))
        self.assertEqual(point_in_mask(m, -0.4, 5), 1)  # redondea a 0


class MascaraTests(unittest.TestCase):
    def test_mascara_booleana_se_acepta(self):
        b = np.zeros((20, 20), bool)
        b[5:15, 5:15] = True
        g = ZoneGeometry("b", b, (10.0, 10.0))
        self.assertEqual(g.contains((10, 10)), 1)
        self.assertEqual(g.contains((1, 1)), 0)
        self.assertEqual(g.perimeter_distance((10, 10)), 0.0)

    def test_varios_contornos_toma_el_mas_cercano(self):
        m = np.zeros((100, 200), np.uint8)
        cv2.rectangle(m, (10, 10), (30, 30), 255, -1)
        cv2.rectangle(m, (150, 10), (170, 30), 255, -1)
        g = ZoneGeometry("dos", m, (90.0, 20.0))
        self.assertAlmostEqual(g.perimeter_distance((120, 20)), 30.0)  # al de la derecha (x=150)
        self.assertAlmostEqual(g.perimeter_distance((50, 20)), 20.0)  # al de la izquierda (x=30)


if __name__ == "__main__":
    unittest.main()