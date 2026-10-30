"""Pruebas de tracking/color_tracker.py.

Ejecutar desde la raíz del proyecto:
    python -m pytest tests/test_color_tracker.py -v
"""

import os
import random
import tempfile
import unittest

import cv2
import numpy as np

from tracking import MASK_FOREGROUND, MODE_COLOR, Detection
from tracking.color_tracker import (
    COLOR_MODE_HSV,
    DEFAULT_PARAMS,
    ColorTracker,
    default_start,
    finalize_params,
    hsv_mask,
    initial_params_from_bbox,
)
from video.reader import VideoReader

DIST_W = 0.003

def _o_apply_morph(mask, open_k=5, close_k=9):
    if open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    if close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask


def _o_pick_component(mask, last_center, min_area, max_area, sol_min=0.0):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    lx, ly = last_center
    best = None
    best_score = 1e18
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        if sol_min > 0:
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 1e-6:
                continue
            solidity = area / hull_area
            if solidity < sol_min:
                continue
        M = cv2.moments(c)
        if M["m00"] <= 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        dist = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
        score = DIST_W * dist
        if score < best_score:
            x, y, w, h = cv2.boundingRect(c)
            best_score = score
            best = (cx, cy, (x, y, w, h))
    return best


def _o_clamp_box(x, y, w, h, W, H):
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(0, min(int(w), W - x))
    h = max(0, min(int(h), H - y))
    return x, y, w, h


def _o_hsv_mask_from_params(roi_bgr, params, include_mask_crop=None):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    hmin = int(params.get("H_MIN", 0))
    hmax = int(params.get("H_MAX", 179))
    smin = int(params.get("S_MIN", 40))
    smax = int(params.get("S_MAX", 255))
    vmin = int(params.get("V_MIN", 40))
    vmax = int(params.get("V_MAX", 255))
    if hmin <= hmax:
        lower = np.array([hmin, smin, vmin], dtype=np.uint8)
        upper = np.array([hmax, smax, vmax], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
    else:
        lower1 = np.array([0, smin, vmin], dtype=np.uint8)
        upper1 = np.array([hmax, smax, vmax], dtype=np.uint8)
        lower2 = np.array([hmin, smin, vmin], dtype=np.uint8)
        upper2 = np.array([179, smax, vmax], dtype=np.uint8)
        mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
    open_k = max(1, int(params.get("OPEN", 3)) | 1)
    close_k = max(1, int(params.get("CLOSE", 5)) | 1)
    mask = _o_apply_morph(mask, open_k=open_k, close_k=close_k)
    if include_mask_crop is not None:
        mask = cv2.bitwise_and(mask, mask, mask=include_mask_crop)
    return mask


def _o_initial_hsv_params_from_bbox(roi_bgr, bbox, include_mask_crop=None):
    x, y, w, h = [int(round(v)) for v in bbox]
    x, y, w, h = _o_clamp_box(x, y, w, h, roi_bgr.shape[1], roi_bgr.shape[0])
    crop = roi_bgr[y:y + h, x:x + w].copy()
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    valid = np.ones(hsv.shape[:2], dtype=bool)
    if include_mask_crop is not None:
        valid_crop = include_mask_crop[y:y + h, x:x + w]
        if valid_crop.shape[:2] == hsv.shape[:2] and cv2.countNonZero(valid_crop) > 0:
            valid = valid_crop > 0
    pix = hsv[valid]
    if pix.size == 0:
        pix = hsv.reshape(-1, 3)
    h = pix[:, 0].astype(np.int32)
    s = pix[:, 1].astype(np.int32)
    v = pix[:, 2].astype(np.int32)
    h_med = int(np.median(h))
    h_margin = 12
    s_lo = max(0, int(np.percentile(s, 10)) - 35)
    s_hi = min(255, int(np.percentile(s, 90)) + 35)
    v_lo = max(0, int(np.percentile(v, 10)) - 35)
    v_hi = min(255, int(np.percentile(v, 90)) + 35)
    return {
        "H_MIN": max(0, h_med - h_margin),
        "H_MAX": min(179, h_med + h_margin),
        "S_MIN": s_lo,
        "S_MAX": s_hi,
        "V_MIN": v_lo,
        "V_MAX": v_hi,
        "OPEN": 3,
        "CLOSE": 5,
        "MINA": 20,
        "MAXA": max(500, int(0.95 * roi_bgr.shape[0] * roi_bgr.shape[1])),
    }


def _o_track_color_marker(roi_bgr, params, last_center, include_mask_crop=None):
    mask = _o_hsv_mask_from_params(roi_bgr, params, include_mask_crop=include_mask_crop)
    pick = _o_pick_component(mask, last_center, params.get("MINA", 20), params.get("MAXA", 120000), sol_min=0.0)
    return pick, mask


def _o_run(rois, params, include_mask, init_center, init_bbox, reinit=None):
    """Rama modo '3' del bucle de main(), literal. reinit: {i: (params, center, bbox)}."""
    reinit = reinit or {}
    last_center, last_bbox = init_center, init_bbox
    out = []
    for i, roi in enumerate(rois):
        if i in reinit:
            params, last_center, last_bbox = reinit[i]
        mask_use = include_mask.copy()
        roi_for_tracker = cv2.bitwise_and(roi, roi, mask=mask_use)
        pick, fg = _o_track_color_marker(roi_for_tracker, params, last_center, include_mask_crop=mask_use)
        if pick is not None:
            cx, cy, (bx, by, bw, bh) = pick
            last_center = (cx, cy)
            last_bbox = (bx, by, bw, bh)
        out.append((pick, fg, last_center, last_bbox))
    return out


# Escenas sintéticas

FW, FH = 320, 240
ROI = (20, 20, 260, 190)  # x0, y0, w0, h0

RED = (0, 0, 255)      # H = 0  -> obliga a un rango circular (H_MIN > H_MAX)
GREEN = (0, 200, 0)    # H ~ 60
BLUE = (255, 60, 0)    # H ~ 115
ORANGE = (0, 128, 255)  # H ~ 15


def make_frames(n, color=GREEN, seed=0, lost=(), distractor=None, twin=False):
    """Fondo gris con ruido + círculo de color que se mueve dentro del ROI."""
    rng = np.random.default_rng(seed)
    frames, truth = [], []
    for i in range(n):
        f = np.clip(rng.normal(105, 8, (FH, FW, 1)), 0, 255).astype(np.uint8).repeat(3, axis=2)
        f = np.clip(f.astype(np.int16) + rng.integers(-3, 4, f.shape), 0, 255).astype(np.uint8)
        cx, cy = 60 + 2.0 * i, 110 + 25 * np.sin(i / 7.0)
        if i not in lost:
            cv2.circle(f, (int(cx), int(cy)), 12, color, -1)
        if distractor is not None:
            cv2.circle(f, (240, 60), 11, distractor, -1)
        if twin:  # otro marcador del MISMO color, fijo y lejos
            cv2.circle(f, (255, 190), 12, color, -1)
        frames.append(f)
        truth.append(None if i in lost else (cx, cy))
    return frames, truth


def crop(frames):
    x0, y0, w0, h0 = ROI
    return [f[y0:y0 + h0, x0:x0 + w0].copy() for f in frames]


def roi_mask(shape="rect"):
    x0, y0, w0, h0 = ROI
    m = np.zeros((h0, w0), np.uint8)
    if shape == "rect":
        m[:] = 255
    else:
        cv2.ellipse(m, (w0 // 2, h0 // 2), (w0 // 2 - 4, h0 // 2 - 4), 0, 0, 360, 255, -1)
    return m


class HsvMaskTests(unittest.TestCase):
    def test_hsv_mask_identica_al_original(self):
        rng = random.Random(1)
        frames, _ = make_frames(8, RED, distractor=BLUE, twin=True)
        rois = crop(frames)
        masks = [None, roi_mask("rect"), roi_mask("elipse")]
        circular = permisivo = 0
        for _ in range(250):
            params = {
                "H_MIN": rng.choice([0, 5, 20, 50, 100, 170, 175]),
                "H_MAX": rng.choice([10, 30, 70, 120, 179, 4]),
                "S_MIN": rng.choice([0, 40, 120, 200]),
                "S_MAX": rng.choice([100, 255]),
                "V_MIN": rng.choice([0, 40, 120]),
                "V_MAX": rng.choice([200, 255]),
                "OPEN": rng.choice([0, 1, 2, 3, 4, 7]),
                "CLOSE": rng.choice([0, 1, 4, 5, 9, 12]),
            }
            for k in rng.sample(list(params), rng.randint(0, 4)):
                del params[k]  # claves que faltan -> valores por defecto
            circular += params.get("H_MIN", 0) > params.get("H_MAX", 179)
            permisivo += params.get("S_MIN", 40) == 0 and params.get("V_MIN", 40) == 0
            roi = rois[rng.randrange(len(rois))]
            m = rng.choice(masks)
            self.assertTrue(
                np.array_equal(hsv_mask(roi, params, m), _o_hsv_mask_from_params(roi, params, m)),
                msg=str(params),
            )
        self.assertGreater(circular, 10)  # la prueba realmente ejercita el rango circular
        self.assertGreater(permisivo, 3)

    def test_enmascarar_el_bgr_antes_de_hsv_importa(self):
        """Justifica el orden del original: BGR enmascarado -> HSV -> morfología -> máscara.

        Un marcador oscuro a ~9 px del borde del ROI, con un rango que acepta lo
        negro: si el BGR se enmascara primero, la zona enmascarada (negra) entra
        en el rango y la morfología (CLOSE) la une al marcador.
        """
        roi = np.full((190, 260, 3), 105, np.uint8)
        m = np.zeros((190, 260), np.uint8)
        cv2.ellipse(m, (130, 95), (126, 91), 0, 0, 360, 255, -1)
        cv2.circle(roi, (18, 95), 5, (10, 10, 10), -1)
        params = {"H_MIN": 0, "H_MAX": 179, "S_MIN": 0, "S_MAX": 255, "V_MIN": 0, "V_MAX": 30,
                  "OPEN": 1, "CLOSE": 15}
        antes = _o_hsv_mask_from_params(cv2.bitwise_and(roi, roi, mask=m), params, m)
        despues = _o_hsv_mask_from_params(roi, params, m)
        self.assertFalse(np.array_equal(antes, despues))  # el orden cambia el resultado
        tr = ColorTracker(m, params)
        tr.reset((18.0, 95.0), (13, 90, 10, 10))
        self.assertTrue(np.array_equal(tr.update(roi).mask, antes))  # el tracker sigue el orden original

    def test_valores_fuera_de_rango_se_acotan_en_lugar_de_fallar(self):
        roi = crop(make_frames(1)[0])[0]
        params = {"H_MIN": -5, "H_MAX": 400, "S_MIN": -1, "S_MAX": 999, "V_MIN": 0, "V_MAX": 255}
        self.assertEqual(hsv_mask(roi, params).shape, roi.shape[:2])


class ParametrosInicialesTests(unittest.TestCase):
    def test_identicos_al_original(self):
        rng = random.Random(2)
        for color in (RED, GREEN, BLUE, ORANGE):
            frames, truth = make_frames(2, color)
            roi = crop(frames)[0]
            cx, cy = truth[0][0] - ROI[0], truth[0][1] - ROI[1]
            masks = [None, roi_mask("rect"), roi_mask("elipse"), np.zeros_like(roi_mask())]
            bboxes = [
                (cx - 14, cy - 14, 28, 28), (cx - 5, cy - 5, 10, 10), (0, 0, 260, 190),
                (-20, -20, 50, 50), (250, 180, 80, 80), (cx, cy, 0, 0), (cx - 14.6, cy - 14.4, 28.5, 27.5),
            ]
            for bbox in bboxes:
                for m in masks:
                    self.assertEqual(
                        initial_params_from_bbox(roi, bbox, m),
                        _o_initial_hsv_params_from_bbox(roi, bbox, m),
                        msg=f"{color} {bbox}",
                    )
        self.assertIsNone(initial_params_from_bbox(roi, (cx, cy, 0, 0)))

    def test_el_rango_propuesto_contiene_al_color(self):
        for color, h_esperado in ((GREEN, 60), (BLUE, 115), (ORANGE, 15)):
            frames, truth = make_frames(2, color)
            roi = crop(frames)[0]
            cx, cy = truth[0][0] - ROI[0], truth[0][1] - ROI[1]
            p = initial_params_from_bbox(roi, (cx - 14, cy - 14, 28, 28), roi_mask())
            self.assertLessEqual(p["H_MIN"], h_esperado + 3)
            self.assertGreaterEqual(p["H_MAX"], h_esperado - 3)

    def test_limitacion_del_original_no_propone_rango_circular_para_rojo(self):
        """Comportamiento heredado: para un rojo puro (H~0) la propuesta no es circular.

        El usuario debe corregirlo con los sliders (H_MIN > H_MAX). Es una
        candidata a mejora; por ahora se conserva idéntico al original.
        """
        frames, truth = make_frames(2, RED)
        roi = crop(frames)[0]
        cx, cy = truth[0][0] - ROI[0], truth[0][1] - ROI[1]
        p = initial_params_from_bbox(roi, (cx - 14, cy - 14, 28, 28), roi_mask())
        self.assertLessEqual(p["H_MIN"], p["H_MAX"])

    def test_finalize_params_replica_lo_que_guardaba_la_calibracion(self):
        # Lo que construía calibrate_color_marker al aceptar (valores de los trackbars):
        original = {"H_MIN": 40, "H_MAX": 80, "S_MIN": 100, "S_MAX": 255, "V_MIN": 50, "V_MAX": 255,
                    "OPEN": max(1, 4 | 1), "CLOSE": max(1, 0 | 1), "MINA": 30, "MAXA": 9000}
        original["COLOR_MODE"] = "HSV"
        crudo = dict(original, OPEN=4, CLOSE=0)
        del crudo["COLOR_MODE"]
        self.assertEqual(finalize_params(crudo), original)
        self.assertEqual(COLOR_MODE_HSV, "HSV")
        self.assertNotIn("COLOR_MODE", crudo)  # no modifica la entrada


class TrackerContraOriginalTests(unittest.TestCase):
    def test_frame_a_frame_identico(self):
        casos = [
            (GREEN, "rect", False), (GREEN, "elipse", True), (RED, "elipse", False),
            (BLUE, "rect", True), (ORANGE, "elipse", True),
        ]
        detecciones = 0
        for color, forma, con_extras in casos:
            frames, _ = make_frames(70, color, lost=set(range(25, 31)) | {50},
                                    distractor=BLUE if color != BLUE else GREEN, twin=con_extras)
            rois = crop(frames)
            m = roi_mask(forma)
            cx0, cy0 = 60 - ROI[0] + 0.0, 110 - ROI[1] + 0.0
            base = initial_params_from_bbox(rois[0], (cx0 - 14, cy0 - 14, 28, 28), m)
            conjuntos = [
                base,
                {**base, "OPEN": 1, "CLOSE": 1},
                {**base, "OPEN": 6, "CLOSE": 12, "MINA": 100, "MAXA": 30000},
                {"H_MIN": 0, "H_MAX": 179, "S_MIN": 0, "S_MAX": 255, "V_MIN": 0, "V_MAX": 255},  # permisivo
                {"H_MIN": 170, "H_MAX": 12, "S_MIN": 80, "S_MAX": 255, "V_MIN": 80, "V_MAX": 255},  # circular
                {},  # todo por defecto
            ]
            for params in conjuntos:
                init_c, init_b = (130.0, 95.0), (120, 85, 20, 20)
                reinit = {45: ({**base, "MINA": 10}, (150.0, 100.0), (140, 90, 20, 20))}
                exp = _o_run(rois, params, m, init_c, init_b, reinit)

                tr = ColorTracker(m, params)
                tr.reset(init_c, init_b)
                for i, roi in enumerate(rois):
                    if i in reinit:
                        tr.params = reinit[i][0]  # recalibración con 'r': cambia params...
                        tr.reset(reinit[i][1], reinit[i][2])  # ...y la posición
                    det = tr.update(roi)
                    pick, fg, lc, lb = exp[i]
                    tag = f"{color} {forma} params={params} frame={i}"
                    self.assertTrue(np.array_equal(det.mask, fg), msg=tag)
                    if pick is None:
                        self.assertFalse(det.found, msg=tag)
                    else:
                        detecciones += 1
                        self.assertEqual((det.center[0], det.center[1], det.bbox), pick, msg=tag)
                    self.assertEqual((tr.last_center, tr.last_bbox), (lc, lb), msg=tag)
        self.assertGreater(detecciones, 800)  # la comparación no pasa en vacío

    def test_arranque_como_main(self):
        """main(): reset(centro del ROI, caja 10 %) y update del frame inicial."""
        frames, truth = make_frames(5, GREEN)
        roi = crop(frames)[0]
        m = roi_mask()
        params = initial_params_from_bbox(roi, (40 - 14, 90 - 14, 28, 28), m)
        c0, b0 = default_start(ROI[2], ROI[3])
        self.assertEqual((c0, b0), ((130.0, 95.0), (0, 0, 26, 19)))

        pick, _ = _o_track_color_marker(cv2.bitwise_and(roi, roi, mask=m), params, c0, include_mask_crop=m)
        tr = ColorTracker(m, params)
        tr.reset(c0, b0)
        det = tr.update(roi)
        self.assertEqual((det.center[0], det.center[1], det.bbox), pick)
        self.assertEqual((tr.last_center, tr.last_bbox), ((pick[0], pick[1]), pick[2]))

    def test_arranque_sin_hallar_el_marcador_conserva_el_inicio_por_defecto(self):
        frames, _ = make_frames(2, GREEN, lost={0})
        roi = crop(frames)[0]
        m = roi_mask()
        tr = ColorTracker(m, {"H_MIN": 50, "H_MAX": 70, "S_MIN": 100, "V_MIN": 100})
        tr.reset(*default_start(ROI[2], ROI[3]))
        self.assertFalse(tr.update(roi).found)
        self.assertEqual((tr.last_center, tr.last_bbox), ((130.0, 95.0), (0, 0, 26, 19)))

    def test_copyto_equivale_a_bitwise_and_en_color(self):
        rng = np.random.default_rng(5)
        img = rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)
        for mask in (roi_mask("elipse")[:60, :80], rng.integers(0, 2, (60, 80)).astype(np.uint8),
                     rng.integers(0, 256, (60, 80), dtype=np.uint8)):
            self.assertTrue(np.array_equal(cv2.copyTo(img, mask), cv2.bitwise_and(img, img, mask=mask)))


class ComportamientoTests(unittest.TestCase):
    def setUp(self):
        frames, self.truth = make_frames(60, GREEN, twin=True)
        self.rois = crop(frames)
        self.mask = roi_mask()
        self.params = initial_params_from_bbox(self.rois[0], (26, 76, 28, 28), self.mask)

    def tracker(self, **kw):
        tr = ColorTracker(self.mask, kw.get("params", self.params))
        tr.reset((40.0, 90.0), (28, 78, 24, 24))
        return tr

    def test_detection_y_modo(self):
        tr = self.tracker()
        det = tr.update(self.rois[10])
        self.assertIsInstance(det, Detection)
        self.assertEqual((det.mask_kind, tr.mode_id), (MASK_FOREGROUND, MODE_COLOR))
        self.assertEqual(det.mask.shape, self.mask.shape)
        self.assertTrue(det.found)

    def test_sigue_al_marcador_cercano_y_no_salta_al_gemelo_lejano(self):
        tr = self.tracker()
        for i in range(0, 60, 1):
            det = tr.update(self.rois[i])
            tx, ty = self.truth[i]
            self.assertAlmostEqual(det.center[0], tx - ROI[0], delta=1.5, msg=f"frame {i}")
            self.assertAlmostEqual(det.center[1], ty - ROI[1], delta=1.5, msg=f"frame {i}")

    def test_params_vacios_usan_los_valores_por_defecto(self):
        tr = ColorTracker(self.mask, None)
        tr.reset((40.0, 90.0), (28, 78, 24, 24))
        det = tr.update(self.rois[3])
        esperado = _o_hsv_mask_from_params(cv2.bitwise_and(self.rois[3], self.rois[3], mask=self.mask), {}, self.mask)
        self.assertTrue(np.array_equal(det.mask, esperado))
        self.assertEqual(DEFAULT_PARAMS["MINA"], 20)

    def test_modificar_params_en_caliente_se_respeta(self):
        tr = self.tracker()
        a = tr.update(self.rois[5]).mask.copy()
        tr.params["S_MIN"] = 250  # prácticamente nada pasa
        b = tr.update(self.rois[5]).mask
        self.assertGreater(int(cv2.countNonZero(a)), int(cv2.countNonZero(b)))

    def test_clave_extra_color_mode_se_ignora(self):
        p = finalize_params(self.params)
        a = ColorTracker(self.mask, p)
        a.reset((40.0, 90.0), (28, 78, 24, 24))
        b = self.tracker()
        self.assertTrue(np.array_equal(a.update(self.rois[8]).mask, b.update(self.rois[8]).mask))

    def test_preview_no_altera_el_estado(self):
        tr = self.tracker()
        antes = (tr.last_center, tr.last_bbox, dict(tr.params))
        det = tr.preview(self.rois[10], {**self.params, "MINA": 5}, (40.0, 90.0))
        self.assertTrue(det.found)
        self.assertEqual((tr.last_center, tr.last_bbox, tr.params), antes)

    def test_preview_equivale_a_update_con_los_mismos_parametros(self):
        a = self.tracker().preview(self.rois[12], self.params, (40.0, 90.0))
        b = self.tracker().update(self.rois[12])
        self.assertEqual((a.center, a.bbox), (b.center, b.bbox))
        self.assertTrue(np.array_equal(a.mask, b.mask))

    def test_update_sin_reset_falla(self):
        with self.assertRaises(RuntimeError):
            ColorTracker(self.mask, self.params).update(self.rois[0])

    def test_no_aplica_filtro_de_solidez(self):
        """El original usa sol_min=0.0 en el modo color: una forma no convexa sí se acepta."""
        roi = np.full((100, 100, 3), 105, np.uint8)
        cv2.rectangle(roi, (10, 10), (60, 20), GREEN, -1)  # forma en L
        cv2.rectangle(roi, (10, 10), (20, 60), GREEN, -1)
        m = np.full((100, 100), 255, np.uint8)
        tr = ColorTracker(m, {"H_MIN": 50, "H_MAX": 70, "S_MIN": 100, "V_MIN": 100, "OPEN": 1, "CLOSE": 1})
        tr.reset((30.0, 30.0), (10, 10, 50, 50))
        self.assertTrue(tr.update(roi).found)


class IntegracionConVideoTests(unittest.TestCase):
    """Video real (MJPG) -> VideoReader -> parámetros iniciales -> tracker."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.results = {}
        for nombre, color in (("rojo", RED), ("verde", GREEN)):
            path = os.path.join(cls.tmp.name, f"{nombre}.avi")
            frames, truth = make_frames(60, color, distractor=BLUE, twin=True)
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (FW, FH))
            for f in frames:
                vw.write(f)
            vw.release()
            cls.results[nombre] = (path, truth)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_extremo_a_extremo(self):
        x0, y0, w0, h0 = ROI
        m = roi_mask("elipse")
        for nombre, (path, truth) in self.results.items():
            with VideoReader(path) as r:
                primero = r.read_at(0)[y0:y0 + h0, x0:x0 + w0]
                cx, cy = truth[0][0] - x0, truth[0][1] - y0
                params = initial_params_from_bbox(primero, (cx - 14, cy - 14, 28, 28), m)
                if nombre == "rojo":
                    # Como haría el usuario con los sliders: rango circular y S/V más exigentes.
                    params.update({"H_MIN": 170, "H_MAX": 10, "S_MIN": 120, "V_MIN": 120})
                tr = ColorTracker(m, finalize_params(params))
                tr.reset((cx + 4, cy + 4), (int(cx) - 8, int(cy) - 8, 20, 20))
                errores = []
                for idx, frame in r.iter_frames(0, 59, 2):
                    det = tr.update(frame[y0:y0 + h0, x0:x0 + w0])
                    self.assertTrue(det.found, msg=f"{nombre} frame {idx}")
                    tx, ty = truth[idx]
                    errores.append(max(abs(det.center[0] - (tx - x0)), abs(det.center[1] - (ty - y0))))
            self.assertLess(max(errores), 3.0, msg=nombre)


if __name__ == "__main__":
    unittest.main()