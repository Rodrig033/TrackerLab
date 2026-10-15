"""Pruebas de ui/polygon_editor.py (lógica pura, sin ventanas).

Ejecutar desde la raíz del proyecto:
    python -m pytest tests/test_polygon_editor.py -v
"""

import random
import unittest

from ui.polygon_editor import (
    MIDPOINT,
    VERTEX,
    Hit,
    PolygonEditor,
    polygon_area,
    polygon_self_intersects,
)

SQUARE = [(100, 100), (300, 100), (300, 300), (100, 300)]


def editor(points=SQUARE, size=(1000, 800)):
    return PolygonEditor(size, points)


class DibujarTests(unittest.TestCase):
    def test_clic_en_espacio_vacio_agrega_al_final(self):
        e = editor(points=[])
        for p in [(10, 10), (200, 10), (200, 150)]:
            self.assertEqual(e.press(*p, scale=1.0), "add")
            e.release()
        self.assertEqual(e.points, [(10, 10), (200, 10), (200, 150)])

    def test_los_puntos_se_redondean_y_se_acotan_a_la_imagen(self):
        e = editor(points=[], size=(100, 50))
        e.press(-20.4, 70.6, 1.0)
        e.release()
        e.press(99.6, 10.4, 1.0)  # redondea a 100 y se acota a 99
        e.release()
        self.assertEqual(e.points, [(0, 49), (99, 10)])

    def test_agregar_y_arrastrar_sin_soltar_es_una_sola_accion(self):
        e = editor(points=[])
        e.press(50, 50, 1.0)
        e.drag_to(80, 90)
        e.drag_to(120, 140)
        e.release()
        self.assertEqual(e.points, [(120, 140)])
        self.assertTrue(e.undo())
        self.assertEqual(e.points, [])  # un solo deshacer quita el vértice completo

    def test_can_accept_pide_tres_puntos(self):
        e = editor(points=[(10, 10), (300, 10)])
        self.assertFalse(e.can_accept)
        e.press(300, 200, 1.0)
        e.release()
        self.assertTrue(e.can_accept)

    def test_un_clic_cerca_de_un_vertice_lo_atrapa_en_lugar_de_agregar_otro(self):
        """Diseño: dentro del radio de captura (12 px de pantalla) manda el vértice existente.
        Para colocar puntos muy juntos hay que acercar el zoom."""
        e = editor(points=[(1, 1), (5, 5)])
        self.assertEqual(e.press(9, 1, 1.0), "vertex")  # a 5.7 px de (5,5)
        e.release()
        self.assertEqual(len(e.points), 2)
        self.assertEqual(e.press(9, 1, 8.0), "add")  # con zoom 8x esos 5.7 px son 45 px de pantalla
        e.release()
        self.assertEqual(len(e.points), 3)


class MoverVerticesTests(unittest.TestCase):
    def test_arrastrar_un_vertice_existente(self):
        e = editor()
        self.assertEqual(e.press(301, 99, 1.0), "vertex")  # a 1.4 px del vértice 1
        self.assertEqual(e.selected, 1)
        e.drag_to(350, 60)
        e.release()
        self.assertEqual(e.points, [(100, 100), (350, 60), (300, 300), (100, 300)])

    def test_el_resto_de_los_vertices_no_se_mueve_y_el_orden_se_conserva(self):
        e = editor()
        e.press(100, 300, 1.0)
        e.drag_to(40, 340)
        e.release()
        self.assertEqual(e.points[:3], SQUARE[:3])
        self.assertEqual(e.points[3], (40, 340))

    def test_un_clic_sin_mover_no_ensucia_el_historial(self):
        e = editor()
        e.press(100, 100, 1.0)
        e.release()
        self.assertFalse(e.undo())  # no hubo ninguna acción que deshacer
        self.assertEqual(e.points, SQUARE)

    def test_arrastre_fuera_de_la_imagen_se_acota(self):
        e = editor(size=(400, 400))
        e.press(300, 300, 1.0)
        e.drag_to(9999, -50)
        e.release()
        self.assertEqual(e.points[2], (399, 0))

    def test_un_arrastre_completo_es_una_sola_accion_deshacible(self):
        e = editor()
        e.press(300, 300, 1.0)
        for step in range(1, 30):
            e.drag_to(300 + step * 5, 300 - step * 2)
        e.release()
        self.assertNotEqual(e.points, SQUARE)
        self.assertTrue(e.undo())
        self.assertEqual(e.points, SQUARE)  # un solo deshacer vuelve al punto de partida
        self.assertTrue(e.redo())
        self.assertEqual(e.points[2], (445, 242))

    def test_drag_to_sin_arrastre_no_hace_nada(self):
        e = editor()
        self.assertFalse(e.drag_to(5, 5))
        self.assertEqual(e.points, SQUARE)

    def test_soltar_dos_veces_es_inofensivo_y_un_press_nuevo_cierra_el_arrastre_olvidado(self):
        e = editor()
        e.press(300, 100, 1.0)
        e.drag_to(320, 80)
        # el botón nunca se "soltó"; llega otro clic en otro vértice
        e.press(300, 300, 1.0)
        self.assertEqual(e.points[1], (320, 80))  # el primer movimiento quedó registrado
        e.release()
        self.assertFalse(e.release())
        self.assertTrue(e.undo())
        self.assertEqual(e.points[1], SQUARE[1])  # y se puede deshacer


class CapturaTests(unittest.TestCase):
    def test_el_radio_se_mide_en_pixeles_de_pantalla(self):
        e = editor()
        # a 8 px de imagen del vértice (100,100)
        self.assertIsNone(e.hit_test(108, 100, scale=2.0))  # a 2x equivale a 16 px de pantalla: fuera
        self.assertEqual(e.hit_test(108, 100, scale=1.0), Hit(VERTEX, 0))  # 8 px: dentro de los 12
        # alejado (scale 0.5): el radio de 12 px de pantalla equivale a 24 px de imagen
        self.assertEqual(e.hit_test(120, 100, scale=0.5), Hit(VERTEX, 0))  # 20 px de imagen = 10 de pantalla
        self.assertIsNone(e.hit_test(130, 100, scale=0.5))  # 30 px de imagen = 15 de pantalla: fuera
        self.assertIsNone(e.hit_test(115, 100, scale=1.0))

    def test_elige_el_vertice_mas_cercano(self):
        e = editor(points=[(100, 100), (110, 100), (300, 300)])
        self.assertEqual(e.hit_test(108, 100, 1.0), Hit(VERTEX, 1))
        self.assertEqual(e.hit_test(103, 100, 1.0), Hit(VERTEX, 0))

    def test_un_vertice_tiene_prioridad_sobre_el_centro_de_un_lado(self):
        e = editor(points=[(100, 100), (120, 100), (110, 160)])
        # el centro del lado 0->1 es (110,100), a 10 px de ambos vértices: gana el vértice
        self.assertEqual(e.hit_test(110, 100, 1.0).kind, VERTEX)

    def test_el_centro_de_un_lado_se_detecta_con_tres_o_mas_vertices(self):
        e = editor()
        self.assertEqual(e.hit_test(200, 100, 1.0), Hit(MIDPOINT, 0))
        self.assertEqual(e.hit_test(100, 200, 1.0), Hit(MIDPOINT, 3))  # lado de cierre 3 -> 0
        two = editor(points=[(100, 100), (300, 100)])
        self.assertEqual(two.midpoints(), [])
        self.assertIsNone(two.hit_test(200, 100, 1.0))

    def test_hover(self):
        e = editor()
        self.assertTrue(e.hover_at(101, 101, 1.0))
        self.assertEqual(e.hover, Hit(VERTEX, 0))
        self.assertFalse(e.hover_at(102, 101, 1.0))  # sin cambio: no hace falta redibujar
        self.assertTrue(e.hover_at(500, 500, 1.0))
        self.assertIsNone(e.hover)


class InsertarYBorrarTests(unittest.TestCase):
    def test_insertar_en_el_centro_de_un_lado(self):
        e = editor()
        self.assertEqual(e.press(200, 100, 1.0), "insert")
        e.drag_to(200, 40)
        e.release()
        self.assertEqual(e.points, [(100, 100), (200, 40), (300, 100), (300, 300), (100, 300)])

    def test_insertar_en_el_lado_de_cierre_agrega_al_final(self):
        e = editor()
        e.press(100, 200, 1.0)  # lado 3 -> 0
        e.release()
        self.assertEqual(e.points, SQUARE + [(100, 200)])

    def test_insertar_se_deshace_de_una_vez(self):
        e = editor()
        e.press(200, 100, 1.0)
        e.drag_to(210, 30)
        e.release()
        self.assertTrue(e.undo())
        self.assertEqual(e.points, SQUARE)

    def test_borrar_el_vertice_seleccionado(self):
        e = editor()
        e.press(300, 100, 1.0)
        e.release()
        self.assertTrue(e.delete_selected())
        self.assertEqual(e.points, [(100, 100), (300, 300), (100, 300)])
        self.assertIsNone(e.selected)
        self.assertFalse(e.delete_selected())  # ya no hay ninguno seleccionado

    def test_borrar_sin_seleccion_no_hace_nada(self):
        e = editor()
        self.assertFalse(e.delete_selected())
        self.assertEqual(e.points, SQUARE)

    def test_borrar_se_puede_deshacer(self):
        e = editor()
        e.press(100, 300, 1.0)
        e.release()
        e.delete_selected()
        e.undo()
        self.assertEqual(e.points, SQUARE)


class DeshacerTests(unittest.TestCase):
    def test_limpiar_todo_se_puede_deshacer(self):
        """El caso que motivó esta mejora: 'c' ya no destruye el trabajo."""
        e = editor()
        self.assertTrue(e.clear())
        self.assertEqual(e.points, [])
        self.assertTrue(e.undo())
        self.assertEqual(e.points, SQUARE)
        self.assertTrue(e.redo())
        self.assertEqual(e.points, [])

    def test_limpiar_un_poligono_vacio_no_registra_nada(self):
        e = editor(points=[])
        self.assertFalse(e.clear())
        self.assertFalse(e.undo())

    def test_deshacer_un_punto_agregado_funciona_como_antes(self):
        e = editor(points=[])
        for p in [(10, 10), (50, 10), (50, 50)]:
            e.press(*p, 1.0)
            e.release()
        e.undo()
        self.assertEqual(e.points, [(10, 10), (50, 10)])

    def test_una_accion_nueva_borra_el_historial_de_rehacer(self):
        e = editor()
        e.press(300, 300, 1.0)
        e.drag_to(350, 350)
        e.release()
        e.undo()
        e.press(500, 500, 1.0)  # agrega un vértice: ya no se puede rehacer el movimiento
        e.release()
        self.assertFalse(e.redo())

    def test_deshacer_durante_un_arrastre_lo_termina_primero(self):
        e = editor()
        e.press(300, 300, 1.0)
        e.drag_to(340, 340)
        self.assertTrue(e.undo())
        self.assertFalse(e.dragging)
        self.assertEqual(e.points, SQUARE)

    def test_deshacer_y_rehacer_sin_historial(self):
        e = editor()
        self.assertFalse(e.undo())
        self.assertFalse(e.redo())

    def test_prueba_aleatoria_deshacer_todo_y_rehacer_todo(self):
        for seed in range(40):
            rng = random.Random(seed)
            e = PolygonEditor((600, 400), [(50, 50), (500, 60), (480, 350), (60, 330)])
            inicio = e.points
            for _ in range(rng.randint(5, 60)):
                accion = rng.choice(["add", "drag", "drag", "insert", "delete", "clear"])
                x, y = rng.uniform(0, 600), rng.uniform(0, 400)
                if accion in ("add", "drag", "insert"):
                    e.press(x, y, rng.choice([0.5, 1.0, 2.0]))
                    for _ in range(rng.randint(0, 4)):
                        e.drag_to(rng.uniform(-50, 650), rng.uniform(-50, 450))
                    e.release()
                elif accion == "delete":
                    if e.points:
                        e.press(*e.points[rng.randrange(len(e.points))], 1.0)
                        e.release()
                    e.delete_selected()
                elif rng.random() < 0.2:
                    e.clear()
                for px, py in e.points:  # los vértices nunca salen de la imagen
                    self.assertTrue(0 <= px <= 599 and 0 <= py <= 399)
            final = e.points
            while e.undo():
                pass
            self.assertEqual(e.points, inicio, msg=f"seed={seed}: deshacer todo no vuelve al inicio")
            while e.redo():
                pass
            self.assertEqual(e.points, final, msg=f"seed={seed}: rehacer todo no vuelve al final")


class GeometriaTests(unittest.TestCase):
    def test_area(self):
        self.assertEqual(polygon_area(SQUARE), 200 * 200)
        self.assertEqual(polygon_area([(0, 0), (10, 0)]), 0.0)
        self.assertEqual(polygon_area([(0, 0), (5, 5), (10, 10)]), 0.0)  # alineados

    def test_autointersecciones(self):
        self.assertFalse(polygon_self_intersects(SQUARE))
        self.assertTrue(polygon_self_intersects([(0, 0), (100, 100), (100, 0), (0, 100)]))  # moño
        concavo = [(0, 0), (100, 0), (100, 100), (50, 40), (0, 100)]  # cóncavo, válido
        self.assertFalse(polygon_self_intersects(concavo))
        self.assertFalse(polygon_self_intersects([(0, 0), (10, 0), (5, 9)]))  # triángulo
        self.assertTrue(polygon_self_intersects([(0, 0), (100, 0), (100, 100), (50, -20), (0, 100)]))  # un lado atraviesa otro

    def test_un_vertice_que_toca_otro_lado_cuenta_como_cruce(self):
        self.assertTrue(polygon_self_intersects([(0, 0), (100, 0), (100, 100), (50, 0), (0, 100)]))

    def test_aviso_para_el_usuario(self):
        self.assertIsNone(editor().warning)
        self.assertIn("cruza", editor(points=[(0, 0), (100, 100), (100, 0), (0, 100)]).warning)
        self.assertIn("alineados", editor(points=[(0, 0), (5, 5), (10, 10)]).warning)
        self.assertIn("alineados", editor(points=[(0, 0), (5, 5), (10, 10), (20, 20)]).warning)  # 4 alineados
        self.assertIn("alineados", editor(points=[(7, 7), (7, 7), (7, 7)]).warning)  # el mismo punto
        # el moño tiene área 0 con la fórmula del cordón, pero NO está alineado: debe decir que se cruza
        self.assertEqual(polygon_area([(0, 0), (100, 100), (100, 0), (0, 100)]), 0.0)
        self.assertIn("cruza", editor(points=[(0, 0), (100, 100), (100, 0), (0, 100)]).warning)
        self.assertTrue(editor(points=[(0, 0), (100, 100), (100, 0), (0, 100)]).warning.isascii())  # cv2.putText solo dibuja ASCII

    def test_bounds_invalidos(self):
        with self.assertRaises(ValueError):
            PolygonEditor((0, 10))


if __name__ == "__main__":
    unittest.main()