"""Pruebas de detection/tecolote.py (los niveles de Tecolote y su preparación para una corrida).

Ejecutar desde la raíz del proyecto:
    python -m pytest tests/test_tecolote.py -v
"""

import unittest

from detection import tecolote as T


class FrecuenciaTests(unittest.TestCase):
    def test_acepta_punto_o_coma_y_espacios(self):
        for texto, v in (("0.5", 0.5), ("0,5", 0.5), (" 1 ", 1.0), (2, 2.0), (0.25, 0.25), ("60", 60.0), ("0.05", 0.05)):
            self.assertEqual(T.parse_every(texto), v, msg=repr(texto))

    def test_rechaza_lo_que_no_sirve(self):
        for malo in ("", "abc", None, "0", "-1", "0.04", "61", "1e9", "nan"):
            with self.assertRaises(ValueError, msg=repr(malo)):
                T.parse_every(malo)

    def test_los_mensajes_son_para_una_persona(self):
        with self.assertRaisesRegex(ValueError, "por ejemplo 0.5"):
            T.parse_every("abc")
        with self.assertRaisesRegex(ValueError, "cada 0.05 a 60 segundos"):
            T.parse_every("0")


class PrepararTests(unittest.TestCase):
    def preparar(self, modo, modelo="/m/rfdetr-small.onnx", every=None, fabrica=None):
        msgs, pedidos = [], []

        def f(ruta):
            pedidos.append(ruta)
            return "CORRECTOR"
        plan = T.prepare(modo, modelo, every, corrector_factory=fabrica or f, out=msgs.append)
        return plan, "\n".join(msgs), pedidos

    def test_apagado(self):
        plan, texto, pedidos = self.preparar(T.APAGADO)
        self.assertEqual((plan.mode, plan.propose_model, plan.corrector, plan.corrects), (T.APAGADO, None, None, False))
        self.assertEqual((texto, pedidos), ("", []))  # ni abre el modelo

    def test_proponer(self):
        plan, texto, pedidos = self.preparar(T.PROPONER)
        self.assertEqual((plan.mode, plan.propose_model, plan.corrector), (T.PROPONER, "/m/rfdetr-small.onnx", None))
        self.assertEqual(pedidos, [])  # proponer no necesita el corrector

    def test_corregir(self):
        plan, texto, pedidos = self.preparar(T.CORREGIR, every="0,25")
        self.assertEqual((plan.mode, plan.propose_model, plan.corrector, plan.every_s, plan.corrects), (T.CORREGIR, "/m/rfdetr-small.onnx", "CORRECTOR", 0.25, True))
        self.assertEqual(pedidos, ["/m/rfdetr-small.onnx"])
        self.assertIn("verifica cada 0.25 s", texto)

    def test_frecuencia_por_omision(self):
        self.assertEqual(self.preparar(T.CORREGIR)[0].every_s, 0.5)

    def test_sin_modelo_configurado_se_trabaja_sin_tecolote_y_se_avisa(self):
        for modo in (T.PROPONER, T.CORREGIR):
            for modelo in (None, ""):
                plan, texto, pedidos = self.preparar(modo, modelo=modelo)
                self.assertEqual((plan.mode, plan.propose_model, plan.corrector, pedidos), (T.APAGADO, None, None, []))
                self.assertIn("no hay modelo configurado", texto)

    def test_si_el_modelo_no_abre_baja_a_proponer(self):
        def falla(ruta):
            raise RuntimeError("archivo dañado")
        plan, texto, _ = self.preparar(T.CORREGIR, fabrica=falla)
        self.assertEqual((plan.mode, plan.propose_model, plan.corrector), (T.PROPONER, "/m/rfdetr-small.onnx", None))
        self.assertIn("Tecolote no podrá corregir (archivo dañado)", texto)

    def test_con_la_fabrica_real_un_modelo_inexistente_tampoco_rompe(self):
        msgs = []
        plan = T.prepare(T.CORREGIR, "/no/existe.onnx", out=msgs.append)
        self.assertEqual(plan.mode, T.PROPONER)
        self.assertIn("No existe el modelo", "\n".join(msgs))

    def test_un_nivel_o_frecuencia_invalidos(self):
        with self.assertRaisesRegex(ValueError, "Nivel de Tecolote desconocido"):
            self.preparar("máximo")
        with self.assertRaises(ValueError):
            self.preparar(T.CORREGIR, every="0")
        with self.assertRaises(ValueError):
            self.preparar(T.APAGADO, every="abc")  # la frecuencia se valida siempre: un error de la ventana no debe pasar inadvertido

    def test_textos_de_la_ventana(self):
        self.assertEqual(T.MODES, ("apagado", "proponer", "corregir"))
        self.assertEqual(set(T.LABELS), set(T.MODES))
        self.assertEqual(set(T.DESCRIPTIONS), set(T.MODES))
        self.assertEqual(T.LABELS[T.CORREGIR], "Proponer y corregir")
        self.assertIn(T.DEFAULT_MODE, T.MODES)


if __name__ == "__main__":
    unittest.main()