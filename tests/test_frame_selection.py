import os
import tempfile
import unittest

from video.frame_selection import (
    BG_N_FRAMES,
    BG_STRIDE,
    FrameRange,
    background_sample_indices,
    compute_timing,
    frame_time_s,
    read_background_frames,
)


# Sirve de oráculo: el código nuevo debe dar exactamente lo mismo.
def _original_bg_indices(center_idx, total, n=BG_N_FRAMES, stride=BG_STRIDE):
    out = []
    start = max(0, center_idx - (n // 2) * stride)
    idx = start
    for _ in range(n):
        if idx >= total:
            break
        out.append(idx)
        idx += stride
    return out


def _original_timing(fps, custom, requested):
    if custom and requested is not None:
        requested = max(1.0, min(float(requested), float(fps)))
        frame_step = max(1, int(round(float(fps) / requested)))
    else:
        requested = float(fps)
        frame_step = 1
    return requested, frame_step, float(fps) / float(frame_step), float(frame_step) / float(fps)


class FrameRangeTests(unittest.TestCase):
    def test_orden_normal(self):
        r = FrameRange.ordered(10, 50)
        self.assertEqual((r.start, r.end, r.was_swapped), (10, 50, False))

    def test_invierte_si_fin_menor_que_inicio(self):
        r = FrameRange.ordered(50, 10)
        self.assertEqual((r.start, r.end, r.was_swapped), (10, 50, True))

    def test_un_solo_frame(self):
        r = FrameRange.ordered(7, 7)
        self.assertEqual(list(r.indices()), [7])

    def test_fin_inclusivo_y_paso(self):
        r = FrameRange(10, 40)
        self.assertEqual(list(r.indices(3)), list(range(10, 41, 3)))
        self.assertEqual(r.count(3), 11)
        self.assertEqual(r.count(), 31)

    def test_validaciones(self):
        with self.assertRaises(ValueError):
            FrameRange(-1, 5)
        with self.assertRaises(ValueError):
            FrameRange(10, 5)
        with self.assertRaises(ValueError):
            FrameRange(0, 5).indices(0)


class TimingTests(unittest.TestCase):
    def test_sin_fps_personalizado(self):
        t = compute_timing(30.0)
        self.assertEqual((t.frame_step, t.effective_fps), (1, 30.0))
        self.assertAlmostEqual(t.dt, 1 / 30)

    def test_fps_personalizado(self):
        t = compute_timing(30.0, custom_fps=True, requested_fps=10)
        self.assertEqual(t.frame_step, 3)
        self.assertAlmostEqual(t.effective_fps, 10.0)
        self.assertAlmostEqual(t.dt, 0.1)

    def test_pedir_mas_que_el_video_se_acota(self):
        t = compute_timing(25.0, custom_fps=True, requested_fps=100)
        self.assertEqual((t.frame_step, t.requested_fps), (1, 25.0))

    def test_pedir_menos_de_uno_se_acota_a_uno(self):
        t = compute_timing(30.0, custom_fps=True, requested_fps=0.2)
        self.assertEqual((t.frame_step, t.requested_fps), (30, 1.0))

    def test_personalizado_sin_valor_equivale_a_original(self):
        self.assertEqual(compute_timing(30.0, True, None).frame_step, 1)

    def test_redondeo_del_original_se_conserva(self):
        # round(2.5) == 2 en Python: pedir 12 FPS en un video de 30 da paso 2.
        self.assertEqual(compute_timing(30.0, True, 12).frame_step, 2)

    def test_fps_invalido(self):
        for bad in (0, -5, float("nan")):
            with self.assertRaises(ValueError):
                compute_timing(bad)

    def test_coincide_con_el_original(self):
        for fps in (24.0, 25.0, 29.97, 30.0, 59.94, 60.0):
            for custom, req in [(False, None), (True, None)] + [(True, r) for r in (0.5, 1, 5, 7.5, 10, 12, 15, 24, 29.97, 30, 60, 120)]:
                t = compute_timing(fps, custom, req)
                exp = _original_timing(fps, custom, req)
                got = (t.requested_fps, t.frame_step, t.effective_fps, t.dt)
                self.assertEqual(got, exp, msg=f"fps={fps} custom={custom} req={req}")

    def test_frame_time(self):
        self.assertAlmostEqual(frame_time_s(130, 100, 30.0), 1.0)
        self.assertEqual(frame_time_s(100, 100, 30.0), 0.0)


class BackgroundIndicesTests(unittest.TestCase):
    def test_ventana_centrada(self):
        idx = background_sample_indices(200, total_frames=1000)
        self.assertEqual(len(idx), BG_N_FRAMES)
        self.assertEqual(idx[0], 200 - (BG_N_FRAMES // 2) * BG_STRIDE)  # 166
        self.assertEqual(idx[BG_N_FRAMES // 2], 200)
        self.assertTrue(all(b - a == BG_STRIDE for a, b in zip(idx, idx[1:])))

    def test_cerca_del_inicio_se_desplaza_hacia_adelante(self):
        idx = background_sample_indices(5, total_frames=1000)
        self.assertEqual(idx[0], 0)
        self.assertEqual(len(idx), BG_N_FRAMES)

    def test_cerca_del_final_se_recorta(self):
        idx = background_sample_indices(990, total_frames=1000)
        self.assertTrue(all(i < 1000 for i in idx))
        self.assertLess(len(idx), BG_N_FRAMES)

    def test_total_desconocido_no_recorta(self):
        self.assertEqual(len(background_sample_indices(100, total_frames=0)), BG_N_FRAMES)

    def test_parametros_invalidos(self):
        with self.assertRaises(ValueError):
            background_sample_indices(10, 100, n=0)
        with self.assertRaises(ValueError):
            background_sample_indices(10, 100, stride=0)

    def test_coincide_con_el_original(self):
        for total in (1, 10, 40, 100, 1000):
            for center in (0, 1, 5, 17, 35, 99, 500, 998, 999, 5000):
                for n, stride in ((35, 2), (10, 1), (7, 5), (1, 1)):
                    got = background_sample_indices(center, total, n, stride)
                    exp = _original_bg_indices(center, total, n, stride)
                    self.assertEqual(got, exp, msg=f"total={total} c={center} n={n} s={stride}")


try:
    import cv2
    import numpy as np

    from video.reader import VideoReader

    HAVE_CV2 = True
except ImportError:  # pragma: no cover
    HAVE_CV2 = False


@unittest.skipUnless(HAVE_CV2, "requiere opencv-python")
class ReadBackgroundFramesTests(unittest.TestCase):
    """Integración con VideoReader usando un video sintético (frame i = gris i)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, "v.avi")
        vw = cv2.VideoWriter(cls.path, cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (64, 48))
        for i in range(100):
            vw.write(np.full((48, 64, 3), i * 2, np.uint8))
        vw.release()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_entrega_los_frames_esperados(self):
        with VideoReader(self.path) as r:
            frames = list(read_background_frames(r, center_idx=50, n=5, stride=3))
            expected = background_sample_indices(50, r.total_frames, 5, 3)
        self.assertEqual(len(frames), 5)
        for f, i in zip(frames, expected):
            self.assertAlmostEqual(float(f.mean()), i * 2, delta=3)  # tolerancia por compresión

    def test_se_detiene_al_final_del_video(self):
        with VideoReader(self.path) as r:
            frames = list(read_background_frames(r, center_idx=98, n=35, stride=2))
        self.assertLess(len(frames), 35)
        self.assertGreater(len(frames), 0)


if __name__ == "__main__":
    unittest.main()