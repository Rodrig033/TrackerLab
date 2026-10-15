import random
import unittest

from analysis import FrameRecord
from analysis.quality import (
    STATUS_DETECTED,
    STATUS_LOST,
    QualityMonitor,
    QualityThresholds,
    assess_quality,
    format_quality,
    frame_status,
    quality_rows,
)

DT = 0.1


def track(spec, start=0):
    """Lista de registros. spec: '.' = detectado (avanza 1 px), 'x' = perdido, 'J' = detectado tras un salto de +100 px."""
    out, x = [], 0.0
    for i, ch in enumerate(spec):
        f = start + i
        if ch == "x":
            out.append(FrameRecord(frame=f, time_s=f * DT, dist_acum_px=0.0))
            continue
        x = x + 100.0 if ch == "J" else x + 1.0
        out.append(FrameRecord(frame=f, time_s=f * DT, dist_acum_px=0.0, cx_abs=x, cy_abs_inv=50.0))
    return out


def lin(n, lost=()):
    """n frames con movimiento lento; los índices en `lost` quedan perdidos."""
    return [FrameRecord(frame=i, time_s=i * DT, dist_acum_px=0.0) if i in lost else
            FrameRecord(frame=i, time_s=i * DT, dist_acum_px=0.0, cx_abs=float(i), cy_abs_inv=10.0) for i in range(n)]


class EstadoTests(unittest.TestCase):
    def test_estado_por_frame(self):
        d, l = track(".x")
        self.assertEqual((frame_status(d), frame_status(l)), (STATUS_DETECTED, STATUS_LOST))


class ConteoYHuecosTests(unittest.TestCase):
    def test_sin_perdidas(self):
        r = assess_quality(lin(100), DT)
        self.assertEqual((r.frames_total, r.frames_detected, r.frames_lost, r.lost_pct), (100, 100, 0, 0.0))
        self.assertEqual((r.gaps, r.n_interruptions, r.longest_gap_frames, r.n_jumps), ((), 0, 0, 0))
        self.assertEqual(r.verdict, "bueno")
        self.assertTrue(r.is_reliable)
        self.assertEqual(r.reasons, ())

    def test_porcentaje_tiempos_y_huecos(self):
        r = assess_quality(lin(100, lost=set(range(10, 15)) | {40} | set(range(60, 63))), DT)
        self.assertEqual((r.frames_lost, r.lost_pct), (9, 9.0))
        self.assertAlmostEqual(r.duration_s, 10.0)
        self.assertAlmostEqual(r.valid_time_s, 9.1)
        self.assertEqual([(g.first_frame, g.last_frame, g.n_frames, g.kind) for g in r.gaps],
                         [(10, 14, 5, "interna"), (40, 40, 1, "interna"), (60, 62, 3, "interna")])
        self.assertAlmostEqual(r.gaps[0].duration_s, 0.5)
        self.assertEqual((r.n_interruptions, r.longest_gap_frames), (3, 5))
        self.assertAlmostEqual(r.longest_gap_s, 0.5)
        self.assertAlmostEqual(r.interruptions_per_min, 3 / (10.0 / 60.0))

    def test_huecos_al_inicio_y_al_final_se_distinguen_de_las_interrupciones(self):
        r = assess_quality(track("xxx.....xx....xxxx"), DT)
        self.assertEqual([(g.kind, g.n_frames) for g in r.gaps], [("inicio", 3), ("interna", 2), ("final", 4)])
        self.assertEqual((r.leading_lost_frames, r.trailing_lost_frames, r.n_interruptions), (3, 4, 1))

    def test_todo_perdido(self):
        r = assess_quality(track("xxxxx"), DT)
        self.assertEqual((r.frames_detected, r.verdict), (0, "deficiente"))
        self.assertIn("ningún frame", r.reasons[0])
        self.assertEqual(r.gaps[0].kind, "inicio")
        self.assertFalse(r.is_reliable)

    def test_sin_registros(self):
        r = assess_quality([], DT)
        self.assertEqual((r.frames_total, r.verdict, r.lost_pct, r.interruptions_per_min), (0, "sin datos", 0.0, 0.0))

    def test_los_frames_se_cuentan_por_registro_no_por_numero_de_frame(self):
        # con frame_step=5 los números de frame saltan de 5 en 5, pero cada registro vale un dt
        regs = [FrameRecord(frame=i * 5, time_s=i * 0.5, dist_acum_px=0.0) if i in (3, 4) else
                FrameRecord(frame=i * 5, time_s=i * 0.5, dist_acum_px=0.0, cx_abs=float(i), cy_abs_inv=1.0) for i in range(10)]
        r = assess_quality(regs, 0.5)
        self.assertEqual((r.gaps[0].first_frame, r.gaps[0].last_frame, r.gaps[0].n_frames), (15, 20, 2))
        self.assertAlmostEqual(r.gaps[0].duration_s, 1.0)


class SaltosTests(unittest.TestCase):
    def test_un_teletransporte_entre_frames_consecutivos_se_marca(self):
        # el recorrido va de x=1 a x=111 (110 px): umbral = max(0.3 * 110, 25) = 33 px; el salto mide 100 px
        r = assess_quality(track("......J....."), DT)
        self.assertEqual(r.n_jumps, 1)
        self.assertEqual(r.jumps[0].frame, 6)
        self.assertAlmostEqual(r.jumps[0].step_px, 100.0)
        self.assertAlmostEqual(r.jump_threshold_px, 33.0, places=6)
        self.assertEqual(r.verdict, "revisar")
        self.assertIn("salto", r.reasons[0])

    def test_reaparecer_lejos_tras_un_hueco_no_es_un_salto(self):
        r = assess_quality(track("...xxxJ..."), DT)
        self.assertEqual(r.n_jumps, 0)  # el sujeto pudo moverse mientras estaba perdido

    def test_movimiento_normal_no_marca_saltos(self):
        r = assess_quality(lin(500), DT)
        self.assertEqual(r.n_jumps, 0)

    def test_el_piso_evita_falsas_alarmas_en_un_sujeto_casi_quieto(self):
        # recorrido de 3 px de diagonal: 30 % serían ~1 px; el piso de 25 px lo impide
        regs = [FrameRecord(frame=i, time_s=i * DT, dist_acum_px=0.0, cx_abs=float(i % 3), cy_abs_inv=0.0) for i in range(30)]
        self.assertEqual(assess_quality(regs, DT).n_jumps, 0)

    def test_umbral_fijo(self):
        r = assess_quality(lin(50), DT, max_step_px=0.5)  # cada paso mide 1 px
        self.assertEqual((r.n_jumps, r.jump_threshold_px), (49, 0.5))

    def test_sin_pasos_no_hay_umbral(self):
        r = assess_quality(track(".x.x."), DT)
        self.assertIsNone(r.jump_threshold_px)


class VeredictoTests(unittest.TestCase):
    def test_umbrales_de_frames_perdidos(self):
        def veredicto(n_lost):
            perdidos = set(range(0, 2 * n_lost, 2))  # alternados: nunca forman un hueco largo
            return assess_quality(lin(100, lost=perdidos), DT).verdict
        self.assertEqual(veredicto(2), "bueno")
        self.assertEqual(veredicto(5), "revisar")
        self.assertEqual(veredicto(14), "revisar")
        self.assertEqual(veredicto(15), "deficiente")

    def test_un_hueco_largo_degrada_aunque_el_porcentaje_sea_bajo(self):
        r = assess_quality(lin(1000, lost=set(range(100, 108))), DT)  # 0.8 % perdido, pero 0.8 s seguidos
        self.assertEqual(r.verdict, "bueno")
        r = assess_quality(lin(1000, lost=set(range(100, 115))), DT)  # 1.5 % perdido, 1.5 s seguidos
        self.assertEqual(r.verdict, "revisar")
        self.assertIn("Hueco más largo", r.reasons[0])
        r = assess_quality(lin(2000, lost=set(range(100, 160))), DT)  # 3 % perdido, 6 s seguidos
        self.assertEqual(r.verdict, "deficiente")

    def test_se_listan_todas_las_razones(self):
        regs = lin(100, lost=set(range(10, 40)))  # 30 % perdido y hueco de 3 s
        r = assess_quality(regs, DT)
        self.assertEqual(r.verdict, "deficiente")
        self.assertEqual(len(r.reasons), 2)

    def test_umbrales_configurables(self):
        estricto = QualityThresholds(lost_pct_warn=0.5, lost_pct_bad=1.0)
        self.assertEqual(assess_quality(lin(100, lost={5, 50}), DT, estricto).verdict, "deficiente")
        self.assertEqual(assess_quality(lin(100, lost={5, 50}), DT).verdict, "bueno")


class StreamingTests(unittest.TestCase):
    def test_streaming_equivale_a_por_lotes(self):
        rng = random.Random(7)
        for _ in range(30):
            n = rng.randint(1, 300)
            regs = lin(n, lost={i for i in range(n) if rng.random() < 0.15})
            for i in rng.sample(range(n), min(3, n)):  # algunos saltos
                if regs[i].found:
                    regs[i] = FrameRecord(frame=i, time_s=i * DT, dist_acum_px=0.0, cx_abs=900.0 + i, cy_abs_inv=500.0)
            mon = QualityMonitor(DT)
            for r in regs:
                mon.update(r)
            self.assertEqual(mon.report(), assess_quality(regs, DT))

    def test_se_puede_pedir_el_informe_varias_veces_sin_alterar_el_estado(self):
        mon = QualityMonitor(DT)
        regs = track("..xx..xx")
        for r in regs[:5]:
            mon.update(r)
        parcial = mon.report()
        self.assertEqual(parcial.gaps[-1].kind, "interna")  # el hueco abierto cuenta, pero aún no terminó
        again = mon.report()
        self.assertEqual(parcial, again)
        for r in regs[5:]:
            mon.update(r)
        self.assertEqual(mon.report(), assess_quality(regs, DT))

    def test_validaciones(self):
        with self.assertRaises(ValueError):
            QualityMonitor(0)


class SalidaTests(unittest.TestCase):
    def test_filas_del_resumen(self):
        r = assess_quality(lin(100, lost={10, 11}), DT)
        rows = dict((a, b) for a, b in quality_rows(r))
        self.assertEqual(rows["calidad_veredicto"], "bueno")
        self.assertEqual((rows["calidad_frames_perdidos"], rows["calidad_frames_perdidos_pct"]), (2, "2.0000"))
        self.assertEqual(rows["calidad_interrupciones"], 1)
        self.assertEqual(rows["calidad_perdidos_al_inicio"], 0)

    def test_texto_para_la_interfaz(self):
        txt = format_quality(assess_quality(lin(100, lost=set(range(10, 40))), DT))
        self.assertIn("DEFICIENTE", txt)
        self.assertIn("70 detectados de 100", txt)
        self.assertIn("Frames perdidos", txt)


class IntegracionConElPipelineTests(unittest.TestCase):
    def test_con_una_corrida_real_en_streaming(self):
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
            frames, _ = BG.make_frames(70, lost=set(range(20, 26)))
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (BG.FW, BG.FH))
            for f in frames:
                vw.write(f)
            vw.release()
            mask = BG.roi_mask("elipse")
            with VideoReader(path) as r:
                bg = compute_background(read_background_frames(r, 30), BG.ROI, mask, False)
                timing = compute_timing(r.fps)
                job = TrackingJob(r, FrameRange(0, 65), timing, BackgroundTracker(mask, bg, DEFAULT_PARAMS), BG.ROI,
                                  start=((40.0, 90.0), (28, 78, 24, 24)))
                monitor = QualityMonitor(timing.dt)
                res = run_tracking(job, on_record=monitor.update, keep_records=True)
            rep = monitor.report()
            self.assertEqual(rep, assess_quality(res.records, timing.dt))
            self.assertEqual(rep.frames_detected, res.frames_found)
            self.assertEqual(rep.frames_total, res.frames_processed)
            self.assertAlmostEqual(rep.valid_time_s, res.kinematic.tiempo_valido_s, places=9)
            self.assertGreaterEqual(rep.n_interruptions, 1)


if __name__ == "__main__":
    unittest.main()