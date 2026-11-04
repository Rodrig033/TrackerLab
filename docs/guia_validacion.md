# Guía: validar el seguimiento de TrackerLab con frames etiquetados a mano

Sirve para medir **qué tan bien sigue TrackerLab al ratón en un video**, comparándolo con cajas que dibujas tú sobre frames sueltos. Con ese mismo conjunto se evaluará después el detector automático.

Todos los comandos se corren **desde la carpeta del proyecto** (`TrackerLab_V1`), con el entorno virtual activo.

---

## 1. Reglas que ya aprendimos (léelas antes de empezar)

1. **Elige al ratón en movimiento.** En la ventana «FRAME MUESTRA» (abre en el primer frame del rango) muévete a un frame donde el ratón esté caminando. Si está quieto más de ~1 segundo en el mismo sitio, el modo fondo lo absorbe y la caja se queda pegada ahí aunque el ratón se vaya. Esto **no lo detecta** el informe de calidad.
2. **Dibuja el ROI holgado**: unos **30 px (medio largo de ratón)** más allá del borde del piso, cubriendo la base de la pared. **No** hasta el borde superior de la pared ni la zona clara de afuera. Si queda justo, el ratón pegado a la pared sale recortado y la caja del tracker sale angosta. En la calibración del modo fondo deja `ROI_SHRINK` en 0.
3. **Para comparar corridas, cambia una sola cosa.** Mismo video, mismo modo, mismo frame de muestra, mismo rango de frames, mismo paso y mismos parámetros. Dale un **nombre de experimento distinto** a cada corrida (por ejemplo `SMR12408_roi_amplio`): así no se sobrescriben y la tabla las distingue.
4. **Las imágenes a etiquetar se sacan de corrido**, igual que las lee el seguimiento (lo hace `sample_frames.py`). Nunca uses capturas hechas con saltos en el video: en un mp4 pueden ser de 1 a 3 frames distintos del que procesó el seguimiento y la evaluación mediría ese desfase.
5. **Verifica el alineamiento antes de evaluar** (`check_frame_alignment.py`). Debe decir `ALINEADO: N de N`.
6. **Etiqueta mirando solo la imagen.** Nada de cajas sugeridas por TrackerLab: sesgan la medición.
7. **Reserva los videos de validación.** No los uses después para entrenar el detector.
8. **Convención de la caja:** ajustada al **cuerpo** visible del ratón, **sin la cola**. Si el ratón no se ve o está tapado casi por completo: etiqueta vacía (`x`). Si dudas: `k`.

---

## 2. Preparación (una sola vez)

1. Copia a tu proyecto, en `tools/`: `sample_frames.py`, `label_frames.py`, `check_frame_alignment.py`, `eval_tracking.py`; y en `tests/`: `test_sample_frames.py`, `test_label_frames.py`, `test_check_frame_alignment.py`, `test_eval_tracking.py`.
2. Corre las pruebas; deben pasar todas: `python -m pytest tests -q`.
3. Verifica que `main.py` tenga `SAVE_RUN_INPUTS = True` (así cada corrida deja una carpeta `entradas_<fecha>`).
4. Crea el archivo de rutas (evita rescribir rutas largas y que las variables se pierdan al abrir otra terminal):

```
cat > /Users/farid/TrackerLab_V1/rutas.sh <<'EOF'
export VAL="/Users/farid/TrackerLab_Resultados/validacion_VIDEO"
export ENTRADAS="/Users/farid/TrackerLab_Resultados/EXPERIMENTO/entradas_FECHA"
EOF
```

Edita las dos rutas con las reales (sin los símbolos `<` `>` ni palabras como `VIDEO` o `FECHA`). En **cada terminal nueva**:

```
cd /Users/farid/TrackerLab_V1
source /Users/farid/TrackerLab_V1/.venv/bin/activate
source /Users/farid/TrackerLab_V1/rutas.sh
echo "$VAL" "$ENTRADAS"
```

---

## 3. Procedimiento paso a paso

### Paso 1 · Corre TrackerLab sobre el video
Siguiendo las reglas 1 a 3. Anota el nombre del experimento y el rango de frames que analizas (debe cubrir buena parte del video: las imágenes se toman de ese rango).

### Paso 2 · Localiza la carpeta de entradas
```
find /Users/farid/TrackerLab_Resultados -type d -name "entradas_*"
```
Imprime una ruta por línea; elige la de tu experimento (la fecha y hora están en el nombre) y ponla en `ENTRADAS` dentro de `rutas.sh`. Para ver qué contiene una corrida:

```
python - "$ENTRADAS" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1] + "/inputs.json"))
print("modo:", d["tracker"]["mode"], "| muestra:", d["sample_idx"], "| rango:", d["frame_range"]["start"], "-", d["frame_range"]["end"], "| paso:", d["timing"]["frame_step"], "| roi_box:", d["roi_box"], "| params:", d["tracker"]["params"])
EOF
```

### Paso 3 · Genera las imágenes a etiquetar
```
python tools/sample_frames.py --entradas "$ENTRADAS" --salida "$VAL" --por-video 20
```
Debe terminar con `Listo: 20 imágenes nuevas...`. Con la misma semilla salen siempre los mismos frames.
- **Varios videos en el mismo conjunto:** repite `--entradas` con cada uno, o agrega después con `--anadir` (no toca lo anterior).
- Si dice que `manifest.json` ya existe, esa carpeta ya tiene un conjunto: **no** uses `--sobrescribir` salvo que quieras rehacer las imágenes.

### Paso 4 · Etiqueta
```
python tools/label_frames.py "$VAL"
```

| Tecla / acción | Qué hace |
|---|---|
| arrastrar el ratón | dibuja la caja (si te equivocas, vuelve a arrastrar: reemplaza) |
| `x` | no se ve el ratón: etiqueta vacía, y pasa a la siguiente |
| `d` o espacio / `a` | siguiente / anterior |
| `n` | próxima imagen pendiente |
| `u` | borra la etiqueta de esa imagen |
| `k` | dudosa: queda sin etiquetar y se anota en `dudosas.txt` |
| `h` | muestra u oculta la ayuda |
| `q` o ESC | salir |

Todo se **guarda al instante**: si cierras la terminal sin `q` no pierdes nada, y al volver a correr el mismo comando continúa en la primera pendiente. Para ver el avance sin abrir la ventana: `python tools/label_frames.py "$VAL" --solo-resumen`. En la barra de arriba debe llegar a `hechas N/N`.

### Paso 5 · Verifica el alineamiento
```
python tools/check_frame_alignment.py "$VAL"
```
- `ALINEADO: N de N` y todos los desfases en `+0`: sigue.
- `ATENCIÓN ... desfases`: las imágenes no son los frames que procesa el seguimiento. No evalúes: regenera el conjunto en una carpeta nueva con el `sample_frames.py` actual y vuelve a etiquetar.

### Paso 6 · Evalúa
```
python tools/eval_tracking.py "$VAL" --entradas "$ENTRADAS"
python tools/eval_tracking.py "$VAL" --entradas "$ENTRADAS" --sin-filtros
```
Puede tardar un rato sin imprimir avance: está repitiendo el seguimiento sobre el video. Escribe en `$VAL/evaluacion_<fecha>/`: `resultados.json`, `por_frame.csv` y `fallos.png` (los peores casos: verde = tu caja, cruz roja = la posición del tracker).

- **Varias corridas** (por ejemplo, antes y después de ampliar el ROI): repite `--entradas`. Cada fila lleva el nombre de su experimento. La fila `TOTAL` mezcla todas; para comparar variantes de un mismo video, ignórala.
- **Varios videos:** pasa los `--entradas` de todos.
- Opciones útiles: `--iou 0.4` (otro umbral), `--hoja-fallos 0` (sin imagen), `--faltantes-sin-raton` (una etiqueta que no existe cuenta como «sin ratón»).

---

## 4. Cómo leer la tabla

| Columna | Qué significa |
|---|---|
| `n`, `con ratón` | frames evaluados, y cuántos tenían ratón |
| `acierto IoU` | la **caja** del tracker se solapa con la tuya al menos 0.5. Muy sensible al tamaño y a la convención (cola) en un animal pequeño |
| `pos. dentro` | la **posición** que queda en el CSV cae dentro de tu caja. **Es la medida principal** |
| `mal ubicado` | encontró algo, pero con IoU < 0.5 (suele ser forma de la caja, no ubicación) |
| `perdido` | no dio nada y el ratón estaba |
| `falso pos.` | se muestra como `falsos / imágenes sin ratón` (por ejemplo `0/0`): dio una posición en una imagen que etiquetaste sin ratón |
| `error centro` | distancia entre la posición y el centro de tu caja, mediana y percentil 90, en px |

- **Sin filtros** = posición cruda; **con filtros** = la que sale en el CSV (suavizado + rechazo de saltos). La diferencia es el costo del suavizado, **no** su beneficio (menos ruido en velocidad y aceleración, que esta evaluación no mide).
- **Cuánto fiarse:** con 20 frames el margen es grande. Un 75 % medido puede ser, con 95 % de confianza, entre ~53 % y ~89 %; un 95 %, entre ~76 % y ~99 %. Sirve para detectar problemas, no para fijar porcentajes exactos.
- `por_frame.csv` trae, frame por frame, tu caja y la del tracker (`x y ancho alto`), el IoU y el error: ahí se ve si un fallo es de ubicación o de forma.

---

## 5. Probar un parámetro sin repetir la corrida a mano

Se copia la carpeta de entradas, se cambia **solo un valor** en la copia y se evalúan las dos juntas:

```
cp -R "$ENTRADAS" "/Users/farid/TrackerLab_Resultados/prueba_sol0"
python - <<'EOF'
import json
p = "/Users/farid/TrackerLab_Resultados/prueba_sol0/inputs.json"
d = json.load(open(p)); d["tracker"]["params"]["SOL"] = 0.0; json.dump(d, open(p, "w"), indent=2)
EOF
python tools/eval_tracking.py "$VAL" --entradas "$ENTRADAS" --entradas "/Users/farid/TrackerLab_Resultados/prueba_sol0" --sin-filtros
```

La tabla tendrá una fila por cada nombre. Parámetros del modo fondo en `inputs.json`: `DIFF`, `OPEN`, `CLOSE`, `MINA`, `MAXA`, `SHRINK`, `SOL`. **No sirve para cambiar el ROI** (la máscara es un archivo): eso requiere una corrida nueva. Con pocos frames, no ajustes parámetros hasta ver varios videos: se sobreajusta a uno.

---

## 6. Errores frecuentes

| Qué ves | Causa | Qué hacer |
|---|---|---|
| `No se encontró inputs.json en <entradas_video1>` | copiaste el comando con el marcador sin sustituir | usa la ruta real de `find`, sin `<` `>` |
| una ruta que empieza con `$/Users/...` | se pegó el `$` de las variables | la ruta no lleva `$` al inicio; las variables se usan como `"$VAL"` |
| `No existe /manifest.json` o `cat: /por_frame.csv` | la variable está vacía (terminal nueva) | `source /Users/farid/TrackerLab_V1/rutas.sh` |
| `Ya existe .../manifest.json. Usa --anadir...` | corriste el comando de **crear** imágenes en vez del de **etiquetar** | corre `label_frames.py` |
| `ATENCIÓN: solo X de N imágenes coinciden` | imágenes de otro instante (sacadas con saltos o de otro rango) | regenera el conjunto con el `sample_frames.py` actual |
| `N frames etiquetados no se procesan` | el rango o el paso de esa corrida no incluye esos frames | corre con el mismo rango/paso, o genera el conjunto con `--entradas` de esa corrida |
| `es de un video que no está en el conjunto` | las entradas son de otro video | usa las entradas del video del conjunto |
| `0 etiquetas que evaluar` | falta etiquetar, o hay archivos de etiqueta con más de una caja | termina de etiquetar; `--solo-resumen` ayuda |
| la ventana del etiquetador no responde a las teclas | no tiene el foco | haz clic sobre ella una vez |

---

## 7. Limitaciones conocidas (modo fondo)

- **Elegir con el ratón quieto** deja la caja pegada a su posición original; el informe de calidad no lo ve.
- **ROI justo** recorta al ratón en paredes y esquinas (medido: los frames donde la caja se salía del ROI tenían IoU promedio 0.48; con el ROI holgado mejoraron 9 de 19 y ninguno empeoró).
- **La cola:** con la cola extendida, el filtro de solidez (0.55) puede **perder** al ratón (confirmado en un frame: con solidez 0 se detecta), y la caja sale más alta que el cuerpo. Con solidez 0 no sabemos si se cuelan falsos positivos.
- **No se miden falsos positivos** mientras el conjunto no tenga imágenes sin ratón.
- **Un solo video, 20 frames:** las cifras de abajo no generalizan todavía.
- **Pendiente de revisar:** si el video se posiciona con un salto inicial impreciso (corridas que no empiezan en el frame 0), los números de frame y los tiempos podrían quedar corridos unos frames por igual.

---

## 8. Resultados de referencia (piloto, video SMR12408, 20 frames)

| | IoU ≥ 0.5 | pos. dentro | mal ubicado | perdido | error centro (mediana / p90) |
|---|---|---|---|---|---|
| Imágenes **desfasadas** (método antiguo), con filtros | 50 % | 65 % | 9 | 1 | 14.0 / 57.4 px |
| Alineadas, con filtros | 75 % | 95 % | 4 | 1 | 12.6 / 28.3 px |
| Alineadas, sin filtros | 75 % | 95 % | 4 | 1 | 5.8 / 12.0 px |
| ROI holgado, sin filtros | 85 % | 95 % | 2 | 1 | 5.8 / 7.2 px |
| Solidez 0, sin filtros | 75 % | 100 % | 5 | 0 | 5.9 / 11.9 px |

La primera fila es la advertencia: con imágenes mal alineadas, los resultados parecían mucho peores de lo que son.

---

## 9. Qué hace cada herramienta

| Herramienta | Para qué |
|---|---|
| `tools/sample_frames.py` | elige frames repartidos por el video y crea `images/`, `manifest.json` y la guía de etiquetado |
| `tools/label_frames.py` | ventana para dibujar las cajas; escribe las etiquetas YOLO en `labels/` |
| `tools/check_frame_alignment.py` | comprueba que las imágenes son los frames que procesa el seguimiento |
| `tools/eval_tracking.py` | repite la corrida y la compara con tus etiquetas |
| `tools/replay.py` | repite una corrida sin ventanas y escribe sus CSV |
| `tools/check_dataset.py` | revisa el conjunto de datos automático (`dataset_<fecha>/`) |