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

## 9. Muchos videos pesados: inventario y reparto de roles

Con videos de 1 GB o más conviene trabajar así:

- **No hace falta procesar los videos completos.** Para entrenar importa la *variedad*, no la cobertura. Elige entre 5 y 8 videos de sesiones distintas y, de cada uno, un tramo de unos minutos con el ratón activo (el rango que ya eliges en TrackerLab).
- **Descarga solo los que vayas a usar, en tandas.** Cuando termine la corrida de un video puedes borrarlo de tu Mac: las carpetas `entradas_` y `dataset_` pesan poco, y lo que va a Colab es el conjunto ya recortado (decenas de MB), no los videos.
- **Cuidado con Drive.** Abrir un video que solo existe «en la nube» puede descargarlo entero. Si no estás seguro de lo que tienes en el disco, empieza con `--sin-abrir`.

**1. Primer vistazo, sin abrir ningún video** (solo nombre, tamaño y fecha). Para no escribir la ruta a mano, escribe `python tools/inventory_videos.py ` (con un espacio al final), arrastra la carpeta a la ventana de Terminal y agrega lo demás:

```
python tools/inventory_videos.py RUTA_DE_LA_CARPETA --sin-abrir --salida inventario
```

**2. Con los videos candidatos ya en tu disco** (una carpeta con los que vas a considerar):

```
python tools/inventory_videos.py RUTA_DE_LA_CARPETA --salida inventario
```

Lee de cada video solo los metadatos y 5 frames repartidos, no lo recorre. Deja `inventario/inventario.csv`, `inventario/hoja.png` (un cuadro por video, con su id y grupo) y `inventario/similitud.csv`. Avisa de los **duplicados** y de las **resoluciones o fps distintos**.

**3. Reparte los roles por grupo, no por video.** `grupo_visual` junta los videos cuyo fondo es casi igual (misma arena y misma cámara). Los videos de un mismo grupo comparten fondo: si unos entrenan y otros validan, la validación sale optimista.

| Rol | Qué videos | Regla |
|---|---|---|
| Entrenamiento | varios grupos, lo más variados posible | `--train` en el armado del conjunto |
| Validación del entrenador | un grupo entero, distinto de los demás | `--valid` |
| **Reservado** para la evaluación final | un grupo entero, distinto de los demás, con tus 20 etiquetas a mano | `--excluir-videos-de` |

El agrupamiento es una ayuda, no una verdad. Si ves videos de la misma arena en grupos distintos (o arenas distintas juntas), ajusta `--umbral` (por omisión 0.9) y mira `hoja.png`. La «fecha» que muestra es la del archivo en tu disco: si lo bajaste de Drive puede ser la de la descarga y no la de la grabación.

**Si todos los videos salen en un solo grupo** (misma arena y cámara), no hay separación por arena posible. Aun así separa por video (idealmente por día o por animal), y cuando reportes resultados dilo así: miden qué tan bien funciona el detector con grabaciones nuevas *en la misma configuración*, no en una configuración nueva.

---

## 10. Revisar las cajas automáticas

Las cajas que deja el seguimiento clásico salen más angostas que las que dibujas tú y a veces incluyen parte de la cola. Si se entrena con ellas tal cual, el detector aprende esos defectos. Esta herramienta te muestra cada imagen **recortada al ROI y con zoom** (el ratón se ve grande) con la caja automática, y tú decides. Se hace una vez por cada video de entrenamiento, antes de armar el conjunto.

```
find /Users/farid/TrackerLab_Resultados -type d -name "dataset_*"
python tools/review_boxes.py "/Users/farid/TrackerLab_Resultados/SMR12408_roi_amplio/dataset_20261103_174940"
```

| Tecla / acción | Qué hace |
|---|---|
| espacio o Enter | la caja está bien: aceptarla y pasar a la siguiente |
| arrastrar el ratón | dibuja la caja correcta (reemplaza a la anterior); luego espacio para pasar a la siguiente |
| `x` | no se ve el ratón o la caja no sirve: **descartar** y pasar a la siguiente |
| `k` | dudosa (no se usará): pasar a la siguiente |
| `u` | deshacer lo decidido en esta imagen (vuelve a la caja automática) |
| `d` / `a` | siguiente / anterior, sin decidir nada |
| `n` | próxima imagen pendiente |
| `h` / `q` o ESC | ayuda / salir |

Todo se guarda **al instante** en `revision.json`, dentro de la carpeta del conjunto. Si cierras de golpe no pierdes nada, y al volver a abrirlo continúa en la primera pendiente. Las etiquetas originales **no se modifican**. Si necesitas que la ventana reciba las teclas, haz clic sobre ella una vez.

**El criterio, el mismo para todas:** caja ajustada al **cuerpo** visible del ratón, **sin la cola**. Ser consistente importa más que acertar al píxel: si unas veces la cola va dentro y otras fuera, el detector no puede aprender una regla. Si dudas si incluir un trozo de cola, déjalo fuera. Descarta (`x`) cuando no hay ratón o la caja está sobre otra cosa, y usa `k` si no estás seguro.

**Cuánto se equivocó la caja automática** (sin abrir la ventana):

```
python tools/review_boxes.py "RUTA_DEL_DATASET" --solo-resumen
```

Imprime cuántas necesitaron corrección y, de esas, el IoU entre la caja automática y la tuya, y cuánto más ancha o alta quedó. Esa es la medida de cuánto sesga el seguimiento clásico.

**Si el conjunto es grande,** no tienes que revisarlo todo: puedes parar y armar con `--solo-revisadas`. Ojo: las imágenes van en orden de frame, así que revisar solo las primeras cubre solo el inicio del tramo. Para un tramo corto (unos 3 minutos) salen unas 180 imágenes, que se revisan en unos 10 minutos.

---

## 11. Armar el conjunto de entrenamiento (detector RF-DETR)

Es otro flujo, aparte de la validación: aquí se juntan las imágenes y cajas **automáticas** de varios videos para entrenar el detector. Para cada video: corre TrackerLab con `COLLECT_DATASET = True` (deja una carpeta `dataset_<fecha>` junto a `entradas_<fecha>`), siguiendo las reglas de la sección 1 (ROI holgado, ratón en movimiento).

Reglas de este flujo:
1. **El video que reservas para evaluar (el de tus etiquetas a mano) nunca entra al entrenamiento.** La herramienta lo comprueba por la huella del video y se niega a armar el conjunto si lo encuentra (`FUGA DE DATOS EVITADA`).
2. **Se reparte por video, no por frame.** Unos videos entrenan y otro valida; si los frames de un mismo video se repartieran entre las dos, la validación saldría optimista.
3. **Se recorta al ROI.** El generador guarda el frame completo y el ratón mide ~65 px; el detector se usará sobre el recorte del ROI, así que se entrena con ese mismo recorte.

Localiza las carpetas y arma el conjunto (sustituye las rutas por las reales que imprima `find`, sin `<` `>`):

```
find /Users/farid/TrackerLab_Resultados -type d -name "dataset_*"
python tools/build_training_set.py \
  --train "/Users/farid/TrackerLab_Resultados/SMR12408_roi_amplio/dataset_20261103_174940" \
  --valid "RUTA_DEL_DATASET_DE_OTRO_VIDEO" \
  --excluir-videos-de "$VAL" \
  --salida /Users/farid/TrackerLab_Resultados/entrenamiento
```

`--train` y `--valid` se pueden repetir (uno por video). Si un conjunto tiene `revision.json` (sección 10), se aplica solo: se usa la caja corregida y se dejan fuera las descartadas y dudosas. Opciones: `--solo-revisadas` (usar únicamente las aceptadas o corregidas), `--margen-roi N` (px extra alrededor del ROI; si lo usas, hay que usar el mismo margen al correr el detector), `--sin-recorte`, `--max-por-video N`, `--sobrescribir`.

Escribe `data.yaml`, `train/` y `valid/` (cada uno con `images` y `labels`) y `conjunto_info.json` con el origen de cada imagen. Lee los `AVISO` que imprime:
- Sin `--valid`, la validación sale de bloques del mismo video: sirve para probar que la tubería funciona, **no** para medir.
- Las etiquetas son automáticas: heredan los errores del seguimiento y su caja más angosta.
- El generador descarta los frames con la caja tocando el borde del ROI: casi no hay ratones pegados a la pared.
- Solo hay imágenes con ratón (ninguna sin ratón).

---

### Con frames etiquetados a mano (cuando el seguimiento falla)

Si en un video el seguimiento clásico falla (reflejos en las paredes, luz que cambia), no sirve como fuente de etiquetas: el generador solo guarda los frames en que el seguimiento acierta, y se pierden justo los casos difíciles que el detector necesita ver. En ese caso las cajas las dibujas tú, y los frames se reparten por **todo** el video sin importar si el seguimiento acertó.

| | Etiquetas automáticas (`--train`, `--valid`) | Etiquetas a mano (`--train-manual`, `--valid-manual`) |
|---|---|---|
| Cuándo | el seguimiento funciona bien en ese video | el seguimiento falla (reflejos, luz) |
| Imágenes que entran | solo donde el seguimiento acertó (y que no toquen el borde del ROI) | repartidas por todo el video |
| Tu trabajo | revisar con `review_boxes.py` | dibujar con `label_frames.py` |

**Procedimiento por video** (el rol de cada uno se decide al armar el conjunto):
1. Corre TrackerLab **solo para definir el ROI**, con `SAVE_RUN_INPUTS = True`. Que el seguimiento salga mal no importa. Dibuja el ROI **holgado** (incluye la base de las paredes, unos 15 a 20 px): así el ratón pegado a la pared no sale cortado y el detector ve también los reflejos. Pon un rango que cubra **todo el video**, para que los frames se repartan por todo él y capturen los cambios de luz.
2. Muestrea. **Una carpeta por rol**; `--entradas` se puede repetir para meter varios videos en la misma:
```
python tools/sample_frames.py --entradas "RUTA_ENTRADAS_R1S2" --entradas "RUTA_ENTRADAS_R2S1" --salida /Users/farid/TrackerLab_Resultados/manual_entrenamiento --por-video 60
python tools/sample_frames.py --entradas "RUTA_ENTRADAS_R3S1" --salida /Users/farid/TrackerLab_Resultados/manual_validacion --por-video 40
```
3. Etiqueta y verifica cada carpeta (debe decir `ALINEADO: N de N`):
```
python tools/label_frames.py /Users/farid/TrackerLab_Resultados/manual_entrenamiento
python tools/check_frame_alignment.py /Users/farid/TrackerLab_Resultados/manual_entrenamiento
```
   **El criterio:** la caja va sobre el **ratón real, nunca sobre su reflejo**, ajustada al cuerpo y sin la cola. Si no estás seguro de cuál es el ratón, `k` (dudosa); si no hay ratón, `x`. Las dudosas y las «sin ratón» no entran al entrenamiento.
   **Audita la consistencia antes de seguir con más videos.** El detector aprende la regla con que dibujas, y lo que más daña es que unas veces la cola vaya dentro de la caja y otras no:
```
python tools/audit_labels.py /Users/farid/TrackerLab_Resultados/manual_entrenamiento
open /Users/farid/TrackerLab_Resultados/manual_entrenamiento/auditoria.png
```
   La hoja muestra un recorte por caja, **todos a la misma escala** y ordenados por área (las más chicas primero). Mira los dos extremos: una caja que incluye la cola se ve más larga que el cuerpo de las demás. El borde rojo es solo una **alerta estadística** (área fuera de lo habitual); un ratón estirado o agachado también sale fuera. Lo que decide es mirar la imagen. Para corregir una caja, abre el etiquetador directamente en ese frame (la propia auditoría imprime el comando): `python tools/label_frames.py "<carpeta>" --ir-a 2757` (o `VIDEO:FRAME` si el conjunto trae varios videos). Arrastra una caja nueva: reemplaza a la anterior. Ojo con la alerta roja: la caja va alineada a los ejes, así que un ratón largo en diagonal da una caja grande aunque esté bien dibujada.
4. **Arma el conjunto.** Hace falta el ROI de cada video, que está en la carpeta `entradas_<fecha>` de una corrida de ese video (el manifiesto de `sample_frames` no lo guarda). Para no teclear ni pegar rutas largas, `find_runs.py` las busca solo (emparejando por la huella del video) y te **imprime el comando exacto**:
```
python tools/find_runs.py --train-manual /Users/farid/TrackerLab_Resultados/manual_entrenamiento --valid-manual /Users/farid/TrackerLab_Resultados/manual_validacion --excluir-videos-de /Users/farid/TrackerLab_Resultados/validacion2
```
   Imprime, para cada video, la corrida que eligió (la más reciente) con su ROI, y avisa si un video tiene varias corridas con ROI distintos. Luego imprime el comando de `build_training_set.py`, con tus rutas reales: **cópialo entero y córrelo**. Si a algún video le falta su corrida, lo dice y no imprime el comando. `--excluir-videos-de` es el conjunto de tu video reservado (SMR12408): impide que se cuele en el entrenamiento. Sin `--train-manual` ni `--valid-manual`, solo lista las corridas que hay.

Se pueden mezclar fuentes automáticas y manuales en el mismo conjunto: la ficha (`conjunto_info.json`) dice de cuál es cada imagen.

---

## 12. Entrenar el detector en Colab

El cuaderno es `colab/entrenar_rfdetr.ipynb`. Se **genera** con `python tools/make_colab_notebook.py` (no se edita a mano), para que la decodificación del ONNX sea siempre la de `learning/onnx_decode.py`.

**Antes de empezar**
1. Comprime la carpeta que armó `build_training_set.py` (la que tiene `data.yaml`, `train/` y `valid/`) en `entrenamiento.zip`. Los videos **nunca** se suben, solo esto (decenas de MB).
2. Abre `colab.research.google.com`, **Archivo → Subir cuaderno**, y elige el `.ipynb`. Luego **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU**.
3. Ejecuta las celdas **de arriba abajo**, una por una. La primera celda de código es la configuración: lo único que normalmente se toca es `MODELO` (`"small"` o `"nano"`).

**Qué mirar en cada paso**
| Paso | Qué comprobar |
|---|---|
| 5 · Revisar el conjunto | Las 8 imágenes: el recorte debe ser la arena, el ratón debe verse grande y la caja verde debe rodear el cuerpo **sin la cola**. Si algo se ve mal, **detente** |
| 6 · Prueba de 2 épocas | Que no falle, el tiempo por época y la memoria de la GPU. Sus mAP no significan nada |
| 8 · Mejor punto | El mAP de validación. Solo sirve para elegir el mejor punto de este entrenamiento |
| 9 · Cajas contra el conjunto | La razón de ancho y alto: por debajo de 1 significa cajas más angostas que las del conjunto |
| 11 · Verificar el ONNX | Debe decir «El ONNX coincide con el modelo original». Si dice ADVERTENCIA, **no uses ese ONNX** |

**Dónde queda el resultado.** Si dejas `GUARDAR_EN_DRIVE = True`, en `Mi unidad/TrackerLab_Entrenamiento/<corrida>/`: `checkpoint_best_total.pth`, el `.onnx`, `run_info.json` (versión de rfdetr, parámetros, tiempos y métricas: **anótala, sirve para reproducir**), `conjunto_info.json` y un `.zip` con todo. Ocupa unos cientos de MB por corrida: el cuaderno imprime el tamaño exacto, y conviene mirar cuánto espacio libre te queda en Drive. El entrenamiento en sí corre en el disco local de Colab (los checkpoints pesan), y se pierde al desconectarte.

**Dos cosas que conviene saber**
- RF-DETR redimensiona cada imagen a un **cuadrado**. Si el recorte del ROI no es cuadrado, la imagen se estira. Es lo que dice la receta oficial del ONNX, y TrackerLab hará lo mismo al usar el modelo, así que entrenamiento y uso son consistentes.
- **La validación de este cuaderno no es la evaluación final.** Sale de videos de validación con etiquetas automáticas o revisadas. La medida que cuenta se hace en tu Mac con el **video reservado** y tus 20 etiquetas a mano.

---

## 13. Evaluar el detector en tu Mac

El cuaderno de Colab deja, en `Mi unidad/TrackerLab_Entrenamiento/<corrida>/`, el mejor modelo, el `.onnx` y `run_info.json`. La evaluación que cuenta se hace **aquí**, sobre tu video reservado, con tus etiquetas a mano.

**Preparación (una vez):**
1. Instala `onnxruntime` en el entorno de TrackerLab: `pip install onnxruntime`.
2. Baja de Drive la carpeta de la corrida a `/Users/farid/TrackerLab_Resultados/modelos/<corrida>/` (el `.onnx` pesa unos 127 MB). **Anota la versión de rfdetr** que dice `run_info.json`.

**Evaluar:**
```
python tools/eval_detector.py /Users/farid/TrackerLab_Resultados/validacion2 --modelo /Users/farid/TrackerLab_Resultados/modelos/<corrida>/inference_model.onnx --entradas RUTA_ENTRADAS_DE_ESE_VIDEO
```
Sustituye `<corrida>` y la ruta de `--entradas` por las reales (`find_runs.py` lista las corridas y sus ROI; para SMR12408 conviene la de ROI holgado). Usa **las mismas métricas y la misma tabla** que `eval_tracking.py`, así que el resultado se compara directamente con el del seguimiento clásico sobre los mismos frames.

- **El detector se usa sobre el recorte del ROI**, que es como se entrenó. Con el mismo margen: si no pasas `--margen-roi`, se lee de las notas que el cuaderno guarda dentro del ONNX (la salida dice de dónde salió).
- Se toma la detección de **mayor confianza**; un frame «sin ratón» con una detección cuenta como falso positivo.
- Imprime el **IoU mediano** (y el del 10 % peor) y cuánto más ancha o alta es la caja del detector que la tuya: por debajo de 1 es más angosta, como pasaba con el seguimiento clásico.
- Escribe `resultados.json`, `por_frame.csv` y `fallos.png` en `evaluacion_detector_<fecha>/`.
- `--modelo` se puede repetir para comparar varios (por ejemplo small y nano) en la misma tabla.

**El tiempo por imagen.** La salida incluye cuánto tarda cada imagen **en esta máquina** (recortar, preparar, correr el modelo y decodificar; sin las dos primeras, que son más lentas). Con `--hilos 1`, `--hilos 2`... se ve cómo cambia. Es el dato que decide cómo se integra el detector en TrackerLab: en Colab salió ~650 ms por imagen en su CPU, pero tu Mac será otro.

**Cómo leer el resultado sobre un video de otro entorno.** Es la prueba honesta de generalización: el detector nunca vio ese entorno. Espera algo peor que en la validación del cuaderno (que era del mismo entorno), y eso es información, no un fallo. Fíjate en `fallos.png` para ver **qué** falla: si el detector marca el reflejo, la sombra o el borde.

---

## 14. Qué hace cada herramienta

| Herramienta | Para qué |
|---|---|
| `tools/sample_frames.py` | elige frames repartidos por el video y crea `images/`, `manifest.json` y la guía de etiquetado |
| `tools/label_frames.py` | ventana para dibujar las cajas; escribe las etiquetas YOLO en `labels/` |
| `tools/check_frame_alignment.py` | comprueba que las imágenes son los frames que procesa el seguimiento |
| `tools/eval_tracking.py` | repite la corrida y la compara con tus etiquetas |
| `tools/review_boxes.py` | revisar y corregir las cajas automáticas de un `dataset_<fecha>` (sección 10) |
| `tools/build_training_set.py` | arma el conjunto de entrenamiento de RF-DETR desde varios `dataset_<fecha>` (etiquetas automáticas) y/o frames etiquetados a mano, recortado al ROI y repartido por video |
| `tools/inventory_videos.py` | inventario de videos pesados: metadatos, duplicados y grupos por arena y cámara (sección 9) |
| `tools/make_colab_notebook.py` | genera el cuaderno de Colab `colab/entrenar_rfdetr.ipynb` (sección 12) |
| `learning/onnx_decode.py` | decodifica la salida del ONNX de RF-DETR (la usan el cuaderno y, después, TrackerLab) |
| `tools/audit_labels.py` | hoja de auditoría de las cajas etiquetadas a mano: un recorte por caja, a la misma escala y ordenado por área (sección 11) |
| `tools/find_runs.py` | localiza las corridas (`entradas_<fecha>`) de los videos de tus conjuntos y te imprime el comando exacto de `build_training_set.py` (sección 11) |
| `tools/eval_detector.py` | evalúa un detector ONNX contra tus frames etiquetados a mano (mismas métricas que `eval_tracking.py`) y mide el tiempo por imagen en tu máquina (sección 13) |
| `tools/replay.py` | repite una corrida sin ventanas y escribe sus CSV |
| `tools/check_dataset.py` | revisa el conjunto de datos automático (`dataset_<fecha>/`) |