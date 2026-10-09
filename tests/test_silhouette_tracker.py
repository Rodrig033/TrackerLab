import os
import random
import tempfile
import unittest

import cv2
import numpy as np

from tracking import MASK_SILHOUETTE, MODE_SILHOUETTE, Detection
from tracking.silhouette_tracker import (
    SilhouetteModel,
    SilhouetteTracker,
    _make_binary_for_subject,
    create_model_from_bbox,
)
from video.reader import VideoReader

def _o_clamp_box(x, y, w, h, W, H):
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(0, min(int(w), W - x))
    h = max(0, min(int(h), H - y))
    return x, y, w, h


def _o_component_candidates_from_mask(mask, min_area=10, max_area=None):
    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for lab in range(1, num):
        area = float(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        w = int(stats[lab, cv2.CC_STAT_WIDTH])
        h = int(stats[lab, cv2.CC_STAT_HEIGHT])
        cx = float(cents[lab][0])
        cy = float(cents[lab][1])
        out.append({"area": area, "bbox": (x, y, w, h), "center": (cx, cy), "mask_bool": labels == lab})
    return out


def _o_make_binary_for_subject(gray, polarity="light", valid_mask=None):
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, m_light = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = m_light if polarity == "light" else cv2.bitwise_not(m_light)
    if valid_mask is not None:
        m = cv2.bitwise_and(m, m, mask=valid_mask)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close, iterations=1)
    return m


def _o_choose_best_initial_component(candidates, box_w, box_h):
    if not candidates:
        return None
    cx_box, cy_box = box_w / 2.0, box_h / 2.0
    box_area = float(box_w * box_h)
    scored = []
    for c in candidates:
        area = c["area"]
        if area < max(20.0, 0.01 * box_area) or area > 0.92 * box_area:
            continue
        cx, cy = c["center"]
        dist = ((cx - cx_box) ** 2 + (cy - cy_box) ** 2) ** 0.5
        score = dist - 0.008 * area
        scored.append((score, c))
    if not scored:
        return None
    scored.sort(key=lambda z: z[0])
    return scored[0][1]


def _o_create_subject_model_from_bbox(roi_bgr, bbox, include_mask_crop=None):
    x, y, w, h = [int(round(v)) for v in bbox]
    x, y, w, h = _o_clamp_box(x, y, w, h, roi_bgr.shape[1], roi_bgr.shape[0])
    if w <= 5 or h <= 5:
        return None
    crop = roi_bgr[y:y + h, x:x + w].copy()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    valid_crop = None
    if include_mask_crop is not None:
        valid_crop = include_mask_crop[y:y + h, x:x + w].copy()
        if valid_crop.shape[:2] != gray.shape[:2] or cv2.countNonZero(valid_crop) <= 0:
            valid_crop = None
    best = None
    best_pol = None
    for pol in ("light", "dark"):
        m = _o_make_binary_for_subject(gray, polarity=pol, valid_mask=valid_crop)
        cand = _o_component_candidates_from_mask(m, min_area=10, max_area=0.95 * w * h)
        chosen = _o_choose_best_initial_component(cand, w, h)
        if chosen is None:
            continue
        cx, cy = chosen["center"]
        dist = ((cx - w / 2.0) ** 2 + (cy - h / 2.0) ** 2) ** 0.5
        score = dist - 0.008 * chosen["area"]
        if best is None or score < best[0]:
            best = (score, chosen)
            best_pol = pol
    if best is None:
        return None
    chosen = best[1]
    bx, by, bw, bh = chosen["bbox"]
    cx_l, cy_l = chosen["center"]
    cx = x + cx_l
    cy = y + cy_l
    bbox_body = (x + bx, y + by, bw, bh)
    sil_full = np.zeros(roi_bgr.shape[:2], dtype=np.uint8)
    sil_full[y:y + h, x:x + w] = (chosen["mask_bool"].astype(np.uint8) * 255)
    return {"polarity": best_pol, "area_ref": float(chosen["area"]), "bbox": bbox_body,
            "center": (float(cx), float(cy)), "last_silhouette": sil_full}


def _o_track_subject_by_silhouette(roi_bgr, model, last_center, last_bbox, include_mask_crop=None):
    if model is None or last_center is None or last_bbox is None:
        return None, None
    H, W = roi_bgr.shape[:2]
    bx, by, bw, bh = [int(round(v)) for v in last_bbox]
    lx, ly = float(last_center[0]), float(last_center[1])
    margin = int(max(50, 1.4 * max(bw, bh)))
    sx1 = max(0, min(int(lx - margin), W - 1))
    sy1 = max(0, min(int(ly - margin), H - 1))
    sx2 = min(W, max(int(lx + margin), sx1 + 2))
    sy2 = min(H, max(int(ly + margin), sy1 + 2))
    search = roi_bgr[sy1:sy2, sx1:sx2].copy()
    if search.size == 0:
        return None, None
    gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    valid_search = None
    if include_mask_crop is not None:
        valid_search = include_mask_crop[sy1:sy2, sx1:sx2].copy()
        if valid_search.shape[:2] != gray.shape[:2] or cv2.countNonZero(valid_search) <= 0:
            valid_search = None
    area_ref = max(float(model.get("area_ref", 20.0)), 20.0)
    min_area = max(12.0, 0.25 * area_ref)
    max_area = min(float(gray.shape[0] * gray.shape[1]) * 0.85, 4.0 * area_ref)
    candidates = []
    used_mask = None
    for pol in (model.get("polarity", "light"), "dark" if model.get("polarity", "light") == "light" else "light"):
        m = _o_make_binary_for_subject(gray, polarity=pol, valid_mask=valid_search)
        cand = _o_component_candidates_from_mask(m, min_area=min_area, max_area=max_area)
        if cand:
            candidates = cand
            used_mask = m
            break
    if not candidates:
        return None, None
    scored = []
    for c in candidates:
        cx_l, cy_l = c["center"]
        cx = sx1 + cx_l
        cy = sy1 + cy_l
        dist = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
        area_penalty = abs(c["area"] - area_ref) / area_ref
        cbx, cby, cbw, cbh = c["bbox"]
        score = dist + 35.0 * area_penalty
        scored.append((score, c, cx, cy))
    scored.sort(key=lambda z: z[0])
    _, chosen, cx, cy = scored[0]
    cbx, cby, cbw, cbh = chosen["bbox"]
    bbox_body = (sx1 + cbx, sy1 + cby, cbw, cbh)
    sil_full = np.zeros(roi_bgr.shape[:2], dtype=np.uint8)
    sil_full[sy1:sy2, sx1:sx2] = (chosen["mask_bool"].astype(np.uint8) * 255)
    model["area_ref"] = 0.95 * area_ref + 0.05 * float(chosen["area"])
    model["bbox"] = bbox_body
    model["center"] = (float(cx), float(cy))
    model["last_silhouette"] = sil_full
    return (float(cx), float(cy), bbox_body), sil_full


def _o_localize_subject_by_model(roi_bgr, model, include_mask_crop=None):
    if model is None:
        return None, None
    H, W = roi_bgr.shape[:2]
    if H <= 2 or W <= 2:
        return None, None
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    valid = None
    if include_mask_crop is not None:
        valid = include_mask_crop.copy()
        if valid.shape[:2] != gray.shape[:2] or cv2.countNonZero(valid) <= 0:
            valid = None
    area_ref = max(float(model.get("area_ref", 20.0)), 20.0)
    min_area = max(12.0, 0.25 * area_ref)
    max_area = min(float(H * W) * 0.85, 4.0 * area_ref)
    candidates_all = []
    preferred_pol = model.get("polarity", "light")
    pols = [preferred_pol, "dark" if preferred_pol == "light" else "light"]
    for pol_i, pol in enumerate(pols):
        m = _o_make_binary_for_subject(gray, polarity=pol, valid_mask=valid)
        cand = _o_component_candidates_from_mask(m, min_area=min_area, max_area=max_area)
        for c in cand:
            area_penalty = abs(c["area"] - area_ref) / area_ref
            bx, by, bw, bh = c["bbox"]
            aspect = max(bw / max(bh, 1), bh / max(bw, 1))
            aspect_penalty = max(0.0, aspect - 5.0)
            pol_penalty = 0.0 if pol_i == 0 else 15.0
            score = 45.0 * area_penalty + 8.0 * aspect_penalty + pol_penalty
            candidates_all.append((score, c, pol))
    if not candidates_all:
        return None, None
    candidates_all.sort(key=lambda z: z[0])
    _, chosen, pol = candidates_all[0]
    bx, by, bw, bh = chosen["bbox"]
    cx, cy = chosen["center"]
    sil_full = np.zeros(roi_bgr.shape[:2], dtype=np.uint8)
    sil_full[:, :] = (chosen["mask_bool"].astype(np.uint8) * 255)
    bbox_body = (bx, by, bw, bh)
    model["polarity"] = pol
    model["area_ref"] = 0.95 * area_ref + 0.05 * float(chosen["area"])
    model["bbox"] = bbox_body
    model["center"] = (float(cx), float(cy))
    model["last_silhouette"] = sil_full
    return (float(cx), float(cy), bbox_body), sil_full


def _o_run(rois, sample_roi, bbox0, include_mask, reinit=None):
    """Rama modo '2' de main(), literal: modelo en el frame muestra, localización
    en el frame de inicio y bucle. reinit: {i: (bbox_nueva)} (tecla 'r')."""
    reinit = reinit or {}
    sample_masked = cv2.bitwise_and(sample_roi, sample_roi, mask=include_mask)
    model = _o_create_subject_model_from_bbox(sample_masked, bbox0, include_mask_crop=include_mask)
    if model is None:
        return None
    roi_start = cv2.bitwise_and(rois[0], rois[0], mask=include_mask)
    initial_pick, _ = _o_localize_subject_by_model(roi_start, model, include_mask_crop=include_mask)
    if initial_pick is not None:
        last_center = (initial_pick[0], initial_pick[1])
        last_bbox = initial_pick[2]
    else:
        last_center = model["center"]
        last_bbox = model["bbox"]
    start = (last_center, last_bbox)
    out = []
    for i, roi in enumerate(rois):
        if i in reinit:
            roi_for_tracker = cv2.bitwise_and(roi, roi, mask=include_mask)
            new_model = _o_create_subject_model_from_bbox(roi_for_tracker, reinit[i], include_mask_crop=include_mask)
            if new_model is not None:
                model = new_model
                last_center = model["center"]
                last_bbox = model["bbox"]
        roi_for_tracker = cv2.bitwise_and(roi, roi, mask=include_mask)
        pick, sil = _o_track_subject_by_silhouette(roi_for_tracker, model, last_center, last_bbox,
                                                   include_mask_crop=include_mask)
        if pick is not None:
            cx, cy, (bx, by, bw, bh) = pick
            last_center = (cx, cy)
            last_bbox = (bx, by, bw, bh)
        out.append((pick, sil, last_center, last_bbox, dict(model, last_silhouette=None)))
    return start, out


# Escenas sintéticas

FW, FH = 360, 260
ROI = (20, 20, 310, 220)  # x0, y0, w0, h0


def make_frames(n, dark=True, seed=0, lost=(), distractor=True, big_distractor=False, start_offset=0):
    """Fondo con ruido + elipse (animal) que se mueve + distractores estáticos."""
    rng = np.random.default_rng(seed)
    bg_v, an_v = (175, 45) if dark else (60, 215)
    frames, truth = [], []
    for i in range(n):
        f = np.clip(rng.normal(bg_v, 5, (FH, FW)), 0, 255).astype(np.uint8)
        f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        t = i + start_offset
        cx, cy = 90 + 2.2 * t, 125 + 30 * np.sin(t / 8.0)
        if i not in lost:
            cv2.ellipse(f, (int(cx), int(cy)), (15, 9), 20, 0, 360, (an_v,) * 3, -1)
        if distractor:
            cv2.circle(f, (270, 70), 9, (an_v,) * 3, -1)  # parecido, estático
        if big_distractor:
            cv2.ellipse(f, (250, 180), (45, 22), 0, 0, 360, (an_v,) * 3, -1)  # mucho más grande
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
        cv2.ellipse(m, (w0 // 2, h0 // 2), (w0 // 2 - 5, h0 // 2 - 5), 0, 0, 360, 255, -1)
    return m


def window_around(truth_pt, half=(26, 20)):
    cx, cy = truth_pt[0] - ROI[0], truth_pt[1] - ROI[1]
    return (cx - half[0], cy - half[1], 2 * half[0], 2 * half[1])


def model_tuple(model):
    """Estado comparable de un modelo (dict original o SilhouetteModel), sin la silueta."""
    d = model if isinstance(model, dict) else {
        "polarity": model.polarity, "area_ref": model.area_ref, "bbox": model.bbox, "center": model.center}
    return (d["polarity"], d["area_ref"], d["bbox"], d["center"])


class FuncionesContraOriginalTests(unittest.TestCase):
    def test_binarizacion_identica(self):
        rng = np.random.default_rng(0)
        for _ in range(40):
            gray = rng.integers(0, 256, (70, 90), dtype=np.uint8)
            cv2.circle(gray, (rng.integers(10, 80), rng.integers(10, 60)), rng.integers(4, 20), int(rng.integers(0, 256)), -1)
            valid = [None, roi_mask("elipse")[:70, :90], np.zeros((70, 90), np.uint8)][rng.integers(0, 3)]
            for pol in ("light", "dark"):
                exp = _o_make_binary_for_subject(gray, pol, valid)
                got = _make_binary_for_subject(gray, pol, valid)
                self.assertTrue(np.array_equal(exp, got))

    def test_modelo_inicial_identico(self):
        rng = random.Random(3)
        for dark in (True, False):
            frames, truth = make_frames(3, dark=dark, big_distractor=True)
            roi = crop(frames)[0]
            for forma in ("rect", "elipse"):
                m = roi_mask(forma)
                masked = cv2.bitwise_and(roi, roi, mask=m)
                ox, oy, ow, oh = window_around(truth[0])
                bboxes = [(ox, oy, ow, oh), (ox + 3.4, oy - 2.6, ow - 5, oh + 7), (0, 0, 310, 220),
                          (-30, -30, 80, 80), (300, 210, 60, 60), (ox, oy, 4, 4), (ox, oy, 5, 40), (200, 130, 90, 60)]
                for bbox in bboxes:
                    exp = _o_create_subject_model_from_bbox(masked, bbox, m)
                    got = create_model_from_bbox(roi, bbox, m)
                    if exp is None:
                        self.assertIsNone(got, msg=str(bbox))
                        continue
                    self.assertEqual(model_tuple(got), model_tuple(exp), msg=str(bbox))
                    self.assertTrue(np.array_equal(got.last_silhouette, exp["last_silhouette"]), msg=str(bbox))

    def test_localizacion_identica(self):
        for dark in (True, False):
            frames, truth = make_frames(30, dark=dark, big_distractor=True)
            rois = crop(frames)
            m = roi_mask("elipse")
            sample = rois[0]
            model_o = _o_create_subject_model_from_bbox(cv2.bitwise_and(sample, sample, mask=m), window_around(truth[0]), m)
            for k in (0, 7, 15, 29):
                mo = dict(model_o)
                tr = SilhouetteTracker(m, create_model_from_bbox(sample, window_around(truth[0]), m))
                pick_o, sil_o = _o_localize_subject_by_model(cv2.bitwise_and(rois[k], rois[k], mask=m), mo, m)
                det = tr.locate(rois[k])
                if pick_o is None:
                    self.assertFalse(det.found)
                    continue
                self.assertEqual((det.center[0], det.center[1], det.bbox), pick_o)
                self.assertTrue(np.array_equal(det.mask, sil_o))
                self.assertEqual(model_tuple(tr.model), model_tuple(mo))


class FlujoCompletoTests(unittest.TestCase):
    def test_frame_a_frame_identico(self):
        casos = [
            dict(dark=True, forma="rect", lost=set(), big=False),
            dict(dark=True, forma="elipse", lost=set(range(30, 36)) | {55}, big=True),
            dict(dark=False, forma="rect", lost=set(range(20, 24)), big=False),
            dict(dark=False, forma="elipse", lost={10, 11, 12, 40}, big=True),
        ]
        encontrados = total = 0
        for cs in casos:
            for start_offset in (0, 25):  # el frame muestra y el de inicio difieren
                frames, truth = make_frames(70, dark=cs["dark"], lost=cs["lost"], big_distractor=cs["big"],
                                            seed=1, start_offset=start_offset)
                rois = crop(frames)
                m = roi_mask(cs["forma"])
                # el modelo se aprende en un frame muestra (donde el animal está en otra posición)
                sample_frames, sample_truth = make_frames(1, dark=cs["dark"], big_distractor=cs["big"], seed=9, start_offset=0)
                sample = crop(sample_frames)[0]
                bbox0 = window_around(sample_truth[0])
                reinit = {45: window_around((truth[45] or truth[44])), 5: (0, 0, 4, 4)}  # una buena y una inválida
                exp = _o_run(rois, sample, bbox0, m, reinit)
                self.assertIsNotNone(exp)
                start_o, out_o = exp

                tr = SilhouetteTracker.from_bbox(m, sample, bbox0)
                self.assertIsNotNone(tr)
                tr.start(rois[0])
                self.assertEqual((tr.last_center, tr.last_bbox), (start_o[0], tuple(int(round(v)) for v in start_o[1])))
                for i, roi in enumerate(rois):
                    if i in reinit:
                        tr.reinit(roi, reinit[i])
                    det = tr.update(roi)
                    pick, sil, lc, lb, model_o = out_o[i]
                    tag = f"{cs} offset={start_offset} frame={i}"
                    total += 1
                    if pick is None:
                        self.assertFalse(det.found, msg=tag)
                        self.assertIsNone(det.mask, msg=tag)
                    else:
                        encontrados += 1
                        self.assertEqual((det.center[0], det.center[1], det.bbox), pick, msg=tag)
                        self.assertTrue(np.array_equal(det.mask, sil), msg=tag)
                        self.assertEqual(det.mask_kind, MASK_SILHOUETTE)
                    self.assertEqual((tr.last_center, tr.last_bbox), (lc, lb), msg=tag)
                    self.assertEqual(model_tuple(tr.model), model_tuple(model_o), msg=tag)
        self.assertGreater(encontrados, 0.7 * total)  # la comparación no pasa en vacío

    def test_enmascarar_solo_la_ventana_equivale_a_enmascarar_todo(self):
        """Justifica la optimización: recortar y luego enmascarar == enmascarar y luego recortar."""
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (220, 310, 3), dtype=np.uint8)
        m = roi_mask("elipse")
        full = cv2.bitwise_and(img, img, mask=m)
        for (y1, y2, x1, x2) in ((0, 100, 0, 100), (50, 170, 100, 260), (120, 220, 210, 310), (90, 92, 5, 7)):
            win = cv2.copyTo(img[y1:y2, x1:x2], m[y1:y2, x1:x2])
            self.assertTrue(np.array_equal(win, full[y1:y2, x1:x2]))

    def test_el_enmascarado_previo_cambia_el_umbral_de_otsu(self):
        """Por qué hay que conservar el orden 'enmascarar -> gris -> Otsu' del original."""
        frames, truth = make_frames(3, dark=True, distractor=False)
        roi = crop(frames)[0]
        m = roi_mask("elipse")
        con = cv2.cvtColor(cv2.bitwise_and(roi, roi, mask=m), cv2.COLOR_BGR2GRAY)
        sin = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        distinto = not np.array_equal(_o_make_binary_for_subject(con, "dark", m), _o_make_binary_for_subject(sin, "dark", m))
        self.assertTrue(distinto)


class ComportamientoTests(unittest.TestCase):
    def setUp(self):
        self.frames, self.truth = make_frames(60, dark=True, seed=2)
        self.rois = crop(self.frames)
        self.mask = roi_mask("elipse")
        self.bbox0 = window_around(self.truth[0])

    def tracker(self):
        tr = SilhouetteTracker.from_bbox(self.mask, self.rois[0], self.bbox0)
        tr.start(self.rois[0])
        return tr

    def test_sigue_al_animal_y_no_al_distractor(self):
        tr = self.tracker()
        for i, roi in enumerate(self.rois):
            det = tr.update(roi)
            self.assertTrue(det.found, msg=f"frame {i}")
            self.assertAlmostEqual(det.center[0], self.truth[i][0] - ROI[0], delta=2.0, msg=f"frame {i}")
            self.assertAlmostEqual(det.center[1], self.truth[i][1] - ROI[1], delta=2.0, msg=f"frame {i}")

    def test_modo_y_tipo_de_mascara(self):
        tr = self.tracker()
        det = tr.update(self.rois[1])
        self.assertIsInstance(det, Detection)
        self.assertEqual((tr.mode_id, det.mask_kind), (MODE_SILHOUETTE, MASK_SILHOUETTE))
        self.assertEqual(det.mask.shape, self.mask.shape)
        self.assertEqual(det.mask.dtype, np.uint8)

    def test_from_bbox_devuelve_none_si_no_hay_silueta(self):
        self.assertIsNone(SilhouetteTracker.from_bbox(self.mask, self.rois[0], (0, 0, 4, 4)))

    def test_reinit_fallido_no_toca_el_estado(self):
        tr = self.tracker()
        tr.update(self.rois[3])
        antes = (tr.last_center, tr.last_bbox, model_tuple(tr.model))
        self.assertFalse(tr.reinit(self.rois[3], (0, 0, 4, 4)))
        self.assertEqual((tr.last_center, tr.last_bbox, model_tuple(tr.model)), antes)

    def test_reinit_exitoso_cambia_el_modelo_y_la_posicion(self):
        tr = self.tracker()
        tr.update(self.rois[3])
        self.assertTrue(tr.reinit(self.rois[40], window_around(self.truth[40])))
        self.assertAlmostEqual(tr.last_center[0], self.truth[40][0] - ROI[0], delta=2.0)

    def test_start_sin_hallar_usa_el_modelo(self):
        """Si no se localiza en el frame de inicio: arranca del modelo y avisa (found=False)."""
        tr = self.tracker()
        tr.model.area_ref = 1e7  # área de referencia absurda: ningún candidato es compatible
        tr.model.center, tr.model.bbox = (50.0, 60.0), (40, 50, 20, 20)
        det = tr.start(self.rois[0])
        self.assertFalse(det.found)
        self.assertIsNotNone(det.mask)  # trae la silueta del modelo
        self.assertEqual((tr.last_center, tr.last_bbox), ((50.0, 60.0), (40, 50, 20, 20)))

    def test_sujeto_perdido_conserva_la_ultima_posicion(self):
        frames, _ = make_frames(40, dark=True, seed=2, lost=set(range(20, 40)), distractor=False)
        rois = crop(frames)
        tr = SilhouetteTracker.from_bbox(self.mask, self.rois[0], self.bbox0)
        tr.start(rois[0])
        for i in range(19):
            tr.update(rois[i])
        ultimo = (tr.last_center, tr.last_bbox)
        perdidos = 0
        for i in range(20, 40):
            det = tr.update(rois[i])
            if not det.found:
                perdidos += 1
                self.assertIsNone(det.mask)
        self.assertGreater(perdidos, 0)

    def test_area_de_referencia_se_suaviza(self):
        tr = self.tracker()
        a0 = tr.model.area_ref
        det = tr.update(self.rois[2])
        area_det = float(np.count_nonzero(det.mask))
        self.assertAlmostEqual(tr.model.area_ref, 0.95 * max(a0, 20.0) + 0.05 * area_det, delta=1.0)

    def test_errores(self):
        tr = self.tracker()
        with self.assertRaises(ValueError):
            tr.update(self.rois[0][:100])  # tamaño distinto al de la máscara
        sin_reset = SilhouetteTracker(self.mask, tr.model)
        with self.assertRaises(RuntimeError):
            sin_reset.update(self.rois[0])
        with self.assertRaises(NotImplementedError):
            tr.preview(self.rois[0], {}, (0, 0))


class LimitacionesHeredadasTests(unittest.TestCase):
    """Comportamientos del original que se conservan a propósito (candidatos a mejora)."""

    def test_sujeto_claro_con_roi_no_rectangular_no_se_localiza(self):
        """Otsu se calcula incluyendo la zona enmascarada (negra): con sujeto claro sobre
        fondo oscuro el umbral cae (~30 en vez de ~131), el fondo queda como un solo
        componente enorme y se descarta. El original hace exactamente lo mismo.
        Si algún día se corrige (Otsu solo sobre píxeles válidos), esta prueba debe invertirse.
        """
        frames, truth = make_frames(12, dark=False, seed=4)
        rois = crop(frames)
        m = roi_mask("elipse")
        bbox = window_around(truth[0])
        model_o = _o_create_subject_model_from_bbox(cv2.bitwise_and(rois[0], rois[0], mask=m), bbox, m)
        pick_o, _ = _o_localize_subject_by_model(cv2.bitwise_and(rois[10], rois[10], mask=m), dict(model_o), m)
        tr = SilhouetteTracker.from_bbox(m, rois[0], bbox)
        self.assertIsNone(pick_o)  # el original no lo localiza...
        self.assertFalse(tr.locate(rois[10]).found)  # ...y el nuevo tampoco (idéntico)
        # con ROI rectangular sí funciona, tanto en el original como en el nuevo
        mr = roi_mask("rect")
        model_r = _o_create_subject_model_from_bbox(cv2.bitwise_and(rois[0], rois[0], mask=mr), bbox, mr)
        self.assertIsNotNone(_o_localize_subject_by_model(cv2.bitwise_and(rois[10], rois[10], mask=mr), dict(model_r), mr)[0])
        self.assertTrue(SilhouetteTracker.from_bbox(mr, rois[0], bbox).locate(rois[10]).found)


class IntegracionConVideoTests(unittest.TestCase):
    """Video real (MJPG) -> VideoReader -> modelo -> localización -> seguimiento."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.casos = {}
        for nombre, dark in (("oscuro", True), ("claro", False)):
            path = os.path.join(cls.tmp.name, f"{nombre}.avi")
            frames, truth = make_frames(60, dark=dark, seed=4)
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (FW, FH))
            for f in frames:
                vw.write(f)
            vw.release()
            cls.casos[nombre] = (path, truth)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_extremo_a_extremo(self):
        x0, y0, w0, h0 = ROI
        for nombre, (path, truth) in self.casos.items():
            # Con sujeto claro y ROI no rectangular el original falla (ver
            # LimitacionesHeredadasTests); aquí se usa ROI rectangular.
            m = roi_mask("elipse" if nombre == "oscuro" else "rect")
            with VideoReader(path) as r:
                sample = r.read_at(0)[y0:y0 + h0, x0:x0 + w0]
                tr = SilhouetteTracker.from_bbox(m, sample, window_around(truth[0]))
                self.assertIsNotNone(tr, nombre)
                start = r.read_at(10)[y0:y0 + h0, x0:x0 + w0]  # frame de inicio distinto al muestra
                self.assertTrue(tr.start(start).found, nombre)
                errores = []
                for idx, frame in r.iter_frames(10, 59, 2):
                    det = tr.update(frame[y0:y0 + h0, x0:x0 + w0])
                    self.assertTrue(det.found, msg=f"{nombre} frame {idx}")
                    tx, ty = truth[idx]
                    errores.append(max(abs(det.center[0] - (tx - x0)), abs(det.center[1] - (ty - y0))))
            self.assertLess(max(errores), 3.0, msg=nombre)


if __name__ == "__main__":
    unittest.main()