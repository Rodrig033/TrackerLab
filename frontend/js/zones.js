
TL.Zones = {
    borrador: [],
    mostrar: true,
    panel: null,
    listado: null,
    resumen: null,

    init() {
        const seccion = TL.$("animal");

        this.panel = TL.crearElemento(
            "div", {className: "panel"}, null
        );

        this.panel.id = "zonas";
        seccion.insertAdjacentElement("afterend", this.panel);

        TL.crearElemento(
            "h2", {text: "Zonas de análisis"}, this.panel
        );

        TL.crearElemento("p", {
            className: "help",
            text: "Dibuja zonas sobre el video para comparar " +
                  "las áreas visitadas por el animal."
        }, this.panel);

        const botones = TL.crearElemento(
            "div", {className: "button-row"}, this.panel
        );

        this.boton(
            botones, "Dibujar zona",
            () => this.activarDibujo(), "primary"
        );

        this.boton(
            botones, "Guardar zona",
            () => this.guardarZona(), "success"
        );

        this.boton(
            botones, "Deshacer punto",
            () => {
                this.borrador.pop();
                TL.Video.dibujarOverlay();
            }
        );

        this.boton(
            botones, "Cancelar dibujo",
            () => this.cancelar()
        );

        this.boton(
            botones, "Mostrar / ocultar",
            () => {
                this.mostrar = !this.mostrar;
                TL.Video.dibujarOverlay();
                TL.Charts.actualizar();
            }
        );

        this.listado = TL.crearElemento(
            "div", {}, this.panel
        );

        this.resumen = TL.crearElemento(
            "div", {}, this.panel
        );

        TL.$("roiCanvas").addEventListener("click", evento => {
            if (TL.modo !== "zona") return;

            const punto = TL.puntoCanvas(
                evento, TL.$("roiCanvas")
            );

            if (!punto) return;

            this.borrador.push(punto);
            TL.Video.dibujarOverlay();
        });

        this.renderizar();
    },

    boton(padre, texto, accion, clase = "") {
        const boton = TL.crearElemento(
            "button",
            {className: `btn ${clase}`, text: texto},
            padre
        );

        boton.type = "button";
        boton.addEventListener("click", accion);

        return boton;
    },

    activarDibujo() {
        if (!TL.videoInfo) {
            alert("Primero carga un video.");
            return;
        }

        TL.modo = "zona";
        this.borrador = [];

        TL.$("video").scrollIntoView({behavior: "smooth"});
    },

    cancelar() {
        this.borrador = [];
        TL.modo = "ninguno";
        TL.Video.dibujarOverlay();
    },

    guardarZona() {
        if (this.borrador.length < 3) {
            alert("Marca al menos tres puntos.");
            return;
        }

        const nombre = prompt(
            "Nombre de la zona:",
            `Zona ${TL.zonas.length + 1}`
        );

        if (nombre === null) return;

        const colores = [
            "#38bdf8", "#f59e0b",
            "#a78bfa", "#fb7185", "#34d399"
        ];

        TL.zonas.push({
            id: crypto.randomUUID(),
            nombre: nombre.trim() ||
                `Zona ${TL.zonas.length + 1}`,
            puntos: this.borrador.map(p => ({...p})),
            color: colores[TL.zonas.length % colores.length],
            visible: true
        });

        this.cancelar();
        this.renderizar();
        this.actualizarEstadisticas();
        TL.Charts.actualizar();
    },

    limpiar(actualizar = true) {
        TL.zonas = [];
        this.borrador = [];

        if (actualizar) {
            this.renderizar();
            TL.Video.dibujarOverlay();
            TL.Charts.actualizar();
        } else if (this.listado) {
            this.renderizar();
        }
    },

    borrar(id) {
        TL.zonas = TL.zonas.filter(z => z.id !== id);
        this.renderizar();
        this.actualizarEstadisticas();
        TL.Video.dibujarOverlay();
        TL.Charts.actualizar();
    },

    renderizar() {
        if (!this.listado) return;

        this.listado.replaceChildren();

        if (!TL.zonas.length) {
            TL.crearElemento("p", {
                className: "help",
                text: "Todavía no hay zonas definidas."
            }, this.listado);
            return;
        }

        TL.zonas.forEach(zona => {
            const fila = TL.crearElemento(
                "div",
                {className: "button-row"},
                this.listado
            );

            const nombre = TL.crearElemento("input", {
                type: "text",
                value: zona.nombre
            }, fila);

            nombre.value = zona.nombre;
            nombre.style.maxWidth = "200px";

            nombre.addEventListener("change", () => {
                zona.nombre = nombre.value.trim() || zona.nombre;
                this.actualizarEstadisticas();
            });

            const color = TL.crearElemento("input", {
                type: "color",
                value: zona.color
            }, fila);

            color.value = zona.color;

            color.addEventListener("input", () => {
                zona.color = color.value;
                TL.Video.dibujarOverlay();
                TL.Charts.actualizar();
            });

            const visible = TL.crearElemento("input", {
                type: "checkbox"
            }, fila);

            visible.checked = zona.visible;

            visible.addEventListener("change", () => {
                zona.visible = visible.checked;
                TL.Video.dibujarOverlay();
                TL.Charts.actualizar();
            });

            this.boton(
                fila, "Eliminar",
                () => this.borrar(zona.id), "danger"
            );
        });
    },

    dibujarPoligono(ctx, puntos, color, cerrar = true) {
        if (!puntos.length) return;

        ctx.beginPath();
        ctx.moveTo(puntos[0].x, puntos[0].y);

        puntos.slice(1).forEach(p => {
            ctx.lineTo(p.x, p.y);
        });

        if (cerrar && puntos.length >= 3) ctx.closePath();

        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.stroke();

        if (cerrar && puntos.length >= 3) {
            ctx.save();
            ctx.globalAlpha = 0.12;
            ctx.fillStyle = color;
            ctx.fill();
            ctx.restore();
        }
    },

    dibujarSobreVideo(ctx) {
        if (this.mostrar) {
            TL.zonas.forEach(zona => {
                if (zona.visible) {
                    this.dibujarPoligono(
                        ctx, zona.puntos, zona.color
                    );
                }
            });
        }

        if (this.borrador.length) {
            this.dibujarPoligono(
                ctx, this.borrador, "#ffffff", false
            );
        }
    },

    dibujarEnTrayectoria(ctx, escala, dx, dy) {
        if (!this.mostrar) return;

        TL.zonas.forEach(zona => {
            if (!zona.visible) return;

            const puntos = zona.puntos.map(p => ({
                x: p.x * escala + dx,
                y: p.y * escala + dy
            }));

            this.dibujarPoligono(ctx, puntos, zona.color);
        });
    },

    contiene(punto, poligono) {
        const [x, y] = punto;
        let dentro = false;

        for (
            let i = 0, j = poligono.length - 1;
            i < poligono.length;
            j = i++
        ) {
            const a = poligono[i];
            const b = poligono[j];

            const cruza =
                (a.y > y) !== (b.y > y) &&
                x < (b.x - a.x) * (y - a.y) /
                    (b.y - a.y) + a.x;

            if (cruza) dentro = !dentro;
        }

        return dentro;
    },

    actualizarEstadisticas() {
        if (!this.resumen) return;

        this.resumen.replaceChildren();

        if (!TL.zonas.length) return;

        TL.crearElemento(
            "h3", {text: "Ocupación por zona"}, this.resumen
        );

        const total = TL.trayectoria.length;

        TL.zonas.forEach(zona => {
            const cantidad = TL.trayectoria.filter(p =>
                this.contiene(p, zona.puntos)
            ).length;

            const porcentaje = total
                ? cantidad / total * 100
                : 0;

            TL.crearElemento("p", {
                text: `${zona.nombre}: ${cantidad} posiciones ` +
                      `(${porcentaje.toFixed(1)}%)`
            }, this.resumen);
        });

        TL.crearElemento("p", {
            className: "help",
            text: "Porcentajes aproximados calculados con " +
                  "los puntos de trayectoria recibidos."
        }, this.resumen);
    }
};
