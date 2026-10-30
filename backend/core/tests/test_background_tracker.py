import os
import random
import tempfile
import unittest

import cv2
import numpy as np

from tracking import MASK_FOREGROUND, MODE_BACKGROUND, Detection
from tracking.background_tracker import (
    DEFAULT_PARAMS,
    BackgroundTracker,
    compute_background,
    foreground_binary,
    normalize_params,
    pick_component,
    shrink_mask,
)
from video.frame_selection import read_background_frames
from video.reader import VideoReader

DIST_W = 0.003

def _o_shrink_mask(mask, px):
    px = int(px)
    if px <= 0:
        return mask
    k = 2 * px + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask, ker, iterations=1)


def _o_preprocess_gray(roi_bgr, include_mask_crop, use_clahe):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=include_mask_crop)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    return gray


def _o_apply_morph(mask, open_k=5, close_k=9):
    if open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    if close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask


def _o_foreground_binary(roi_bgr, bg_gray, include_mask_crop, diff_thr, open_k, close_k,
                         use_clahe, apply_bw, polarity):
    gray = _o_preprocess_gray(roi_bgr, include_mask_crop, use_clahe)
    diff = cv2.absdiff(gray, bg_gray)
    _, fg = cv2.threshold(diff, int(diff_thr), 255, cv2.THRESH_BINARY)
    if apply_bw:
        thr = 160
        if polarity == "white_on_black":
            _, bw = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
        else:
            _, bw = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
        fg = cv2.bitwise_and(fg, bw)
    open_k = int(open_k) | 1
    close_k = int(close_k) | 1
    fg = _o_apply_morph(fg, open_k=open_k, close_k=close_k)
    fg = cv2.bitwise_and(fg, fg, mask=include_mask_crop)
    return fg


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


def _o_bg_median(rois, include_mask_crop, use_clahe):
    frames = [_o_preprocess_gray(r, include_mask_crop, use_clahe) for r in rois]
    if not frames:
        return None
    stack = np.stack(frames, axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def _o_track(rois, bg_gray, include_mask, params, polarity, apply_bw, use_clahe,
             init_center, init_bbox, reinit=None):
    """Rama modo '1' del bucle de main(), literal. reinit: {i: (center, bbox)}."""
    reinit = reinit or {}
    last_center, last_bbox = init_center, init_bbox
    out = []
    for i, roi in enumerate(rois):
        if i in reinit:
            last_center, last_bbox = reinit[i]
        mask_use = _o_shrink_mask(include_mask, params.get("SHRINK", 0))
        fg = _o_foreground_binary(roi, bg_gray, mask_use, params["DIFF"], params["OPEN"],
                                  params["CLOSE"], use_clahe=use_clahe, apply_bw=apply_bw,
                                  polarity=polarity)
        pick = _o_pick_component(fg, last_center, params["MINA"], params["MAXA"],
                                 sol_min=params.get("SOL", 0.0))
        if pick is not None:
            cx, cy, (bx, by, bw, bh) = pick
            last_center = (cx, cy)
            last_bbox = (bx, by, bw, bh)
        out.append((pick, fg, last_center, last_bbox))
    return out



# Escenas sintéticas

FW, FH = 320, 240
ROI = (20, 20, 260, 190)  # x0, y0, w0, h0


def make_background(seed=0):
    rng = np.random.default_rng(seed)
    gx = np.linspace(70, 120, FW, dtype=np.float32)[None, :].repeat(FH, 0)
    base = np.stack([gx, gx * 0.9, gx * 1.1], axis=2)
    base += rng.normal(0, 1.5, base.shape)
    return np.clip(base, 0, 255).astype(np.uint8)


def make_frames(n, seed=0, lost=(), second_blob=False, dark=False):
    """Fondo + círculo brillante que se mueve; `lost` = índices sin sujeto."""
    rng = np.random.default_rng(seed + 1)
    bg = make_background(seed)
    frames, centers = [], []
    for i in range(n):
        f = bg.astype(np.int16) + rng.normal(0, 2.0, bg.shape).astype(np.int16)
        f = np.clip(f, 0, 255).astype(np.uint8)
        cx, cy = 60 + 2.0 * i, 110 + 25 * np.sin(i / 7.0)  # se mantiene dentro del ROI (x 20..280)
        if i not in lost:
            cv2.circle(f, (int(cx), int(cy)), 13, (15, 15, 15) if dark else (230, 230, 230), -1)
        if second_blob:
            cv2.circle(f, (250, 60), 11, (235, 235, 235), -1)
        frames.append(f)
        centers.append(None if i in lost else (cx, cy))
    return frames, centers


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


PARAM_SETS = [
    dict(DEFAULT_PARAMS),
    {"DIFF": 25, "OPEN": 4, "CLOSE": 10, "MINA": 100, "MAXA": 20000, "SHRINK": 6, "SOL": 0.55},
    {"DIFF": 12, "OPEN": 1, "CLOSE": 1, "MINA": 50, "MAXA": 90000},  # sin SHRINK ni SOL
    {"DIFF": 18, "OPEN": 7, "CLOSE": 15, "MINA": 200, "MAXA": 80000, "SHRINK": 3, "SOL": 0.8},
]

class CoincideConElOriginalTests(unittest.TestCase):
    def test_foreground_binary_identico(self):
        rng = random.Random(1)
        frames, _ = make_frames(6)
        bg = np.median(np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]), axis=0).astype(np.uint8)
        for _ in range(60):
            fr = crop([frames[rng.randrange(6)]])[0]
            m = roi_mask(rng.choice(["rect", "elipse"]))
            kw = dict(
                diff_thr=rng.choice([5, 18, 40, 200]),
                open_k=rng.choice([1, 2, 3, 4, 5, 8, 12]),
                close_k=rng.choice([1, 2, 3, 6, 9, 14]),
                use_clahe=rng.random() < 0.5,
                apply_bw=rng.random() < 0.5,
                polarity=rng.choice(["white_on_black", "black_on_white"]),
            )
            bg_roi = crop([cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)])[0][:, :, 0] if bg.shape != m.shape else bg
            exp = _o_foreground_binary(fr, bg_roi, m, **kw)
            got = foreground_binary(fr, bg_roi, m, **kw)
            self.assertTrue(np.array_equal(exp, got), msg=str(kw))

    def test_foreground_binary_con_clahe_reutilizado_es_identico(self):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        frames, _ = make_frames(8)
        rois = crop(frames)
        m = roi_mask()
        bg = _o_bg_median(rois, m, True)
        for r in rois:  # el mismo objeto CLAHE en muchas llamadas no cambia nada
            exp = _o_foreground_binary(r, bg, m, 18, 5, 9, True, False, "white_on_black")
            got = foreground_binary(r, bg, m, 18, 5, 9, True, False, "white_on_black", clahe=clahe)
            self.assertTrue(np.array_equal(exp, got))

    def test_pick_component_identico(self):
        rng = random.Random(2)
        for _ in range(300):
            m = np.zeros((190, 260), np.uint8)
            for _ in range(rng.randint(0, 5)):
                x, y = rng.randint(10, 240), rng.randint(10, 170)
                if rng.random() < 0.5:
                    cv2.circle(m, (x, y), rng.randint(2, 25), 255, -1)
                else:
                    cv2.rectangle(m, (x, y), (x + rng.randint(3, 30), y + rng.randint(1, 4)), 255, -1)
            lc = (rng.uniform(0, 260), rng.uniform(0, 190))
            args = (rng.choice([0, 20, 200]), rng.choice([500, 80000]), rng.choice([0.0, 0.55, 0.9]))
            self.assertEqual(pick_component(m, lc, *args), _o_pick_component(m, lc, *args))

    def test_fondo_identico_con_y_sin_clahe_y_pocos_frames(self):
        frames, _ = make_frames(40)
        rois = crop(frames)
        m = roi_mask("elipse")
        for use_clahe in (False, True):
            for n in (1, 2, 5, 34, 35):
                got = compute_background(iter(frames[:n]), ROI, m, use_clahe)
                exp = _o_bg_median(rois[:n], m, use_clahe)
                self.assertTrue(np.array_equal(got, exp), msg=f"clahe={use_clahe} n={n}")
        self.assertIsNone(compute_background(iter([]), ROI, m, False))

    def test_apply_mask_equivale_a_bitwise_and_con_mascara(self):
        """copyTo debe dar lo mismo que bitwise_and(mask=) con CUALQUIER valor no nulo."""
        from tracking.background_tracker import _apply_mask
        rng = np.random.default_rng(3)
        img = rng.integers(0, 256, (90, 120), dtype=np.uint8)
        for mask in (
            roi_mask("elipse")[:90, :120],                              # 0 / 255
            (rng.integers(0, 2, (90, 120)) * 1).astype(np.uint8),        # 0 / 1
            rng.integers(0, 256, (90, 120), dtype=np.uint8),             # valores arbitrarios
        ):
            self.assertTrue(np.array_equal(_apply_mask(img, mask), cv2.bitwise_and(img, img, mask=mask)))

    def test_shrink_identico(self):
        m = roi_mask("elipse")
        for px in (-3, 0, 1, 5, 12):
            self.assertTrue(np.array_equal(shrink_mask(m, px), _o_shrink_mask(m, px)))

    def test_tracker_completo_frame_a_frame(self):
        frames, _ = make_frames(80, lost=set(range(30, 36)) | {55}, second_blob=True)
        rois = crop(frames)
        for mshape in ("rect", "elipse"):
            m = roi_mask(mshape)
            for use_clahe, apply_bw, pol in [(False, False, "white_on_black"),
                                             (True, True, "white_on_black"),
                                             (False, True, "black_on_white")]:
                bg = _o_bg_median(rois[:35], m, use_clahe)
                for params in PARAM_SETS:
                    init_c, init_b = (80.0 - 20 + 0, 110.0 - 20), (60, 70, 26, 26)
                    reinit = {50: ((200.0, 90.0), (190, 80, 20, 20))}
                    exp = _o_track(rois, bg, m, params, pol, apply_bw, use_clahe, init_c, init_b, reinit)

                    tr = BackgroundTracker(m, bg, params, polarity=pol, apply_bw=apply_bw, use_clahe=use_clahe)
                    tr.reset(init_c, init_b)
                    for i, roi in enumerate(rois):
                        if i in reinit:
                            tr.reset(*reinit[i])
                        det = tr.update(roi)
                        pick, fg, lc, lb = exp[i]
                        tag = f"{mshape} clahe={use_clahe} bw={apply_bw} {pol} {params} frame={i}"
                        self.assertTrue(np.array_equal(det.mask, fg), msg=tag)
                        if pick is None:
                            self.assertFalse(det.found, msg=tag)
                        else:
                            self.assertEqual((det.center[0], det.center[1], det.bbox), pick, msg=tag)
                        self.assertEqual(tr.last_center, lc, msg=tag)
                        self.assertEqual(tr.last_bbox, lb, msg=tag)

    def test_tracker_completo_sujeto_oscuro_polaridad_negra(self):
        """Ejercita de verdad la rama black_on_white + apply_bw (sujeto oscuro)."""
        frames, _ = make_frames(60, lost={20, 21}, dark=True)
        rois = crop(frames)
        m = roi_mask("elipse")
        bg = _o_bg_median(rois[:35], m, False)
        detectados = 0
        for apply_bw in (False, True):
            for pol in ("white_on_black", "black_on_white"):
                params = PARAM_SETS[0]
                exp = _o_track(rois, bg, m, params, pol, apply_bw, False, (40.0, 90.0), (27, 77, 26, 26))
                tr = BackgroundTracker(m, bg, params, polarity=pol, apply_bw=apply_bw)
                tr.reset((40.0, 90.0), (27, 77, 26, 26))
                for i, roi in enumerate(rois):
                    det = tr.update(roi)
                    pick, fg, lc, lb = exp[i]
                    self.assertTrue(np.array_equal(det.mask, fg), msg=f"{pol} bw={apply_bw} frame={i}")
                    self.assertEqual(det.found, pick is not None)
                    if pick is not None:
                        self.assertEqual((det.center[0], det.center[1], det.bbox), pick)
                        detectados += 1
        # black_on_white + bw detecta al sujeto oscuro; white_on_black + bw no
        self.assertGreater(detectados, 100)

    def test_la_comparacion_no_es_vacia(self):
        """El tracker debe encontrar al sujeto en la mayoría de frames y perderlo en los vacíos."""
        frames, truth = make_frames(80, lost=set(range(30, 36)))
        rois = crop(frames)
        m = roi_mask()
        bg = compute_background(iter(frames[:35]), ROI, m, False)
        tr = BackgroundTracker(m, bg, DEFAULT_PARAMS)
        tr.reset((60.0, 90.0), (47, 77, 26, 26))
        found = lost = 0
        for i, roi in enumerate(rois):
            det = tr.update(roi)
            if truth[i] is None:
                lost += not det.found
            else:
                self.assertTrue(det.found, msg=f"frame {i}")
                # coordenadas del recorte = absolutas - (x0, y0)
                self.assertAlmostEqual(det.center[0], truth[i][0] - ROI[0], delta=2.0)
                self.assertAlmostEqual(det.center[1], truth[i][1] - ROI[1], delta=2.0)
                found += 1
        self.assertGreater(found, 60)
        self.assertEqual(lost, 6)


class ComportamientoTests(unittest.TestCase):
    def setUp(self):
        self.frames, self.truth = make_frames(60)
        self.rois = crop(self.frames)
        self.mask = roi_mask()
        self.bg = compute_background(iter(self.frames[:35]), ROI, self.mask, False)

    def tracker(self, **kw):
        params = kw.pop("params", DEFAULT_PARAMS)
        tr = BackgroundTracker(self.mask, self.bg, params, **kw)
        tr.reset((60.0, 90.0), (47, 77, 26, 26))
        return tr

    def test_detection_es_foreground_y_modo_1(self):
        tr = self.tracker()
        det = tr.update(self.rois[10])
        self.assertIsInstance(det, Detection)
        self.assertEqual(det.mask_kind, MASK_FOREGROUND)
        self.assertEqual(tr.mode_id, MODE_BACKGROUND)
        self.assertEqual(det.mask.shape, self.mask.shape)

    def test_sujeto_perdido_conserva_ultima_posicion_y_trae_mascara(self):
        tr = self.tracker()
        tr.update(self.rois[5])
        last = (tr.last_center, tr.last_bbox)
        det = tr.update(self.bg_as_frame())
        self.assertFalse(det.found)
        self.assertIsNotNone(det.mask)
        self.assertEqual((tr.last_center, tr.last_bbox), last)

    def bg_as_frame(self):
        return cv2.cvtColor(self.bg, cv2.COLOR_GRAY2BGR)

    def test_elige_el_componente_mas_cercano(self):
        frames, _ = make_frames(50, second_blob=True)
        rois = crop(frames)
        tr = BackgroundTracker(self.mask, self.bg, DEFAULT_PARAMS)
        tr.reset((60.0, 90.0), (47, 77, 26, 26))  # cerca del sujeto móvil, lejos del fijo
        det = tr.update(rois[10])
        self.assertLess(det.center[0], 100)  # no saltó al círculo fijo (x ~ 230)

    def test_update_sin_reset_falla(self):
        tr = BackgroundTracker(self.mask, self.bg, DEFAULT_PARAMS)
        with self.assertRaises(RuntimeError):
            tr.update(self.rois[0])

    def test_preview_no_altera_el_estado(self):
        tr = self.tracker()
        antes = (tr.last_center, tr.last_bbox, dict(tr.params))
        det = tr.preview(self.rois[10], {"DIFF": 30, "OPEN": 3, "CLOSE": 5, "MINA": 10, "MAXA": 50000}, (60.0, 90.0))
        self.assertTrue(det.found)
        self.assertEqual((tr.last_center, tr.last_bbox, tr.params), antes)

    def test_preview_equivale_a_update_con_los_mismos_parametros(self):
        p = {"DIFF": 22, "OPEN": 5, "CLOSE": 7, "MINA": 100, "MAXA": 60000, "SHRINK": 4, "SOL": 0.5}
        a = self.tracker(params=p).preview(self.rois[12], p, (60.0, 90.0))
        b = self.tracker(params=p).update(self.rois[12])
        self.assertEqual((a.center, a.bbox), (b.center, b.bbox))
        self.assertTrue(np.array_equal(a.mask, b.mask))

    def test_normalize_params(self):
        p = normalize_params({"DIFF": 1, "OPEN": 1, "CLOSE": 1, "MINA": 1, "MAXA": 2})
        self.assertEqual((p["SHRINK"], p["SOL"]), (0, 0.0))  # SOL=0.0, NO 0.55
        self.assertEqual(DEFAULT_PARAMS["SOL"], 0.55)
        with self.assertRaises(ValueError):
            normalize_params({"DIFF": 1})
        with self.assertRaises(ValueError):
            BackgroundTracker(self.mask, self.bg, {"DIFF": 1})

    def test_normalize_no_modifica_el_original(self):
        p = {"DIFF": 1, "OPEN": 1, "CLOSE": 1, "MINA": 1, "MAXA": 2}
        normalize_params(p)
        self.assertNotIn("SOL", p)

    def test_set_background(self):
        tr = self.tracker()
        nuevo = self.bg.copy()
        tr.set_background(nuevo)
        self.assertIs(tr.bg_gray, nuevo)
        for malo in (None, np.zeros((5, 5), np.uint8), self.bg.astype(np.float32)):
            with self.assertRaises(ValueError):
                tr.set_background(malo)

    def test_mascara_erosionada_se_calcula_una_vez(self):
        tr = self.tracker(params={**DEFAULT_PARAMS, "SHRINK": 5})
        tr.update(self.rois[3])
        primera = tr._shrunk[1]
        tr.update(self.rois[4])
        self.assertIs(tr._shrunk[1], primera)  # mismo objeto: no se recalculó
        tr.params["SHRINK"] = 8  # si cambia el parámetro, se recalcula
        tr.update(self.rois[5])
        self.assertEqual(tr._shrunk[0], 8)
        self.assertTrue(np.array_equal(tr._shrunk[1], _o_shrink_mask(self.mask, 8)))


class IntegracionConVideoTests(unittest.TestCase):
    """Video real (MJPG) -> VideoReader -> fondo -> tracker."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, "v.avi")
        frames, cls.truth = make_frames(70)
        vw = cv2.VideoWriter(cls.path, cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (FW, FH))
        for f in frames:
            vw.write(f)
        vw.release()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_extremo_a_extremo(self):
        m = roi_mask()
        x0, y0, w0, h0 = ROI
        with VideoReader(self.path) as r:
            bg = compute_background(read_background_frames(r, 35), ROI, m, False)
            self.assertEqual(bg.shape, m.shape)
            self.assertAlmostEqual(float(bg.mean()), float(make_background().mean(axis=2)[y0:y0 + h0, x0:x0 + w0].mean()), delta=6)
            tr = BackgroundTracker(m, bg, DEFAULT_PARAMS)
            tr.reset((60.0, 90.0), (47, 77, 26, 26))
            errores = []
            for idx, frame in r.iter_frames(0, 69, 2):
                det = tr.update(frame[y0:y0 + h0, x0:x0 + w0])
                self.assertTrue(det.found, msg=f"frame {idx}")
                tx, ty = self.truth[idx]
                errores.append(max(abs(det.center[0] - (tx - x0)), abs(det.center[1] - (ty - y0))))
        self.assertLess(max(errores), 3.0)


if __name__ == "__main__":
    unittest.main()