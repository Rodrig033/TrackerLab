
TL.ROI = {
    historial: [],
    futuros: [],
    seleccionado: -1,
    arrastrando: -1,
    origenArrastre: null,
    movimiento: false,
    hover: null,
    maxHistorial: 200,

    init() {
        const canvas = TL.$("roiCanvas");

        TL.$("modoRoi").addEventListener("click", () => {
            TL.modo = "roi";
            this.informar("Modo edición activo. Arrastra los nodos.");
            TL.$("video").scrollIntoView({behavior: "smooth"});
        });

        TL.$("deshacer").addEventListener(
            "click", () => this.deshacer()
        );

        TL.$("limpiar").addEventListener(
            "click", () => this.limpiar()
        );

        TL.$("guardarRoi").addEventListener(
            "click", () => this.guardar()
        );

        canvas.addEventListener(
            "pointerdown", e => this.presionar(e)
        );
        canvas.addEventListener(
            "pointermove", e => this.mover(e)
        );
        canvas.addEventListener(
            "pointerup", e => this.soltar(e)
        );
        canvas.addEventListener(
            "pointercancel", e => this.soltar(e)
        );
        canvas.addEventListener("lostpointercapture", () => {
            this.terminarArrastre();
        });
        canvas.addEventListener("pointerleave", () => {
            if (this.arrastrando < 0) {
                this.hover = null;
                TL.Video.dibujarOverlay();
            }
        });

        document.addEventListener("keydown", e => {
            if (TL.modo !== "roi") return;

            const etiqueta = document.activeElement?.tagName;
            if (["INPUT", "TEXTAREA", "SELECT"].includes(etiqueta)) {
                return;
            }

            if (e.ctrlKey && e.key.toLowerCase() === "z") {
                e.preventDefault();
                if (e.shiftKey) this.rehacer();
                else this.deshacer();
            } else if (
                e.ctrlKey && e.key.toLowerCase() === "y"
            ) {
                e.preventDefault();
                this.rehacer();
            } else if (
                e.key === "Delete" ||
                e.key === "Backspace"
            ) {
                e.preventDefault();
                this.eliminarSeleccionado();
            } else if (e.key === "Escape") {
                TL.modo = "ninguno";
                this.terminarArrastre();
                this.informar("Edición desactivada.");
            }
        });

        this.agregarControles();
    },

    agregarControles() {
        const guardar = TL.$("guardarRoi");

        const rehacer = document.createElement("button");
        rehacer.type = "button";
        rehacer.className = "btn";
        rehacer.textContent = "Rehacer";
        rehacer.addEventListener(
            "click", () => this.rehacer()
        );
        guardar.parentElement.appendChild(rehacer);

        const eliminar = document.createElement("button");
        eliminar.type = "button";
        eliminar.className = "btn";
        eliminar.textContent = "Eliminar nodo";
        eliminar.addEventListener(
            "click", () => this.eliminarSeleccionado()
        );
        guardar.parentElement.appendChild(eliminar);
    },

    reiniciarEditor() {
        this.historial = [];
        this.futuros = [];
        this.seleccionado = -1;
        this.arrastrando = -1;
        this.origenArrastre = null;
        this.movimiento = false;
        this.hover = null;
    },

    copia() {
        return TL.roi.map(p => ({x: p.x, y: p.y}));
    },

    iguales(a, b) {
        return JSON.stringify(a) === JSON.stringify(b);
    },

    registrar(antes) {
        this.historial.push(antes);

        if (this.historial.length > this.maxHistorial) {
            this.historial.shift();
        }

        this.futuros = [];
    },

    cambiado() {
        TL.roiGuardada = false;
        TL.calibracionGuardada = false;
        TL.Video.dibujarOverlay();
        this.informar();
    },

    escala() {
        const canvas = TL.$("roiCanvas");
        const rect = canvas.getBoundingClientRect();

        return Math.min(
            rect.width / Math.max(1, canvas.width),
            rect.height / Math.max(1, canvas.height)
        );
    },

    distancia(a, b) {
        return Math.hypot(a.x - b.x, a.y - b.y);
    },

    puntosMedios() {
        if (TL.roi.length < 3) return [];

        return TL.roi.map((p, i) => {
            const siguiente = TL.roi[(i + 1) % TL.roi.length];

            return {
                x: (p.x + siguiente.x) / 2,
                y: (p.y + siguiente.y) / 2,
                lado: i
            };
        });
    },

    buscarNodo(punto) {
        const escala = Math.max(this.escala(), 0.000001);
        const toleranciaVertice = 12 / escala;
        const toleranciaMedio = 9 / escala;

        let mejor = null;
        let distanciaMenor = Infinity;

        TL.roi.forEach((p, i) => {
            const distancia = this.distancia(punto, p);

            if (
                distancia <= toleranciaVertice &&
                distancia < distanciaMenor
            ) {
                mejor = {tipo: "vertice", indice: i};
                distanciaMenor = distancia;
            }
        });

        if (mejor) return mejor;

        distanciaMenor = Infinity;

        this.puntosMedios().forEach(p => {
            const distancia = this.distancia(punto, p);

            if (
                distancia <= toleranciaMedio &&
                distancia < distanciaMenor
            ) {
                mejor = {tipo: "medio", indice: p.lado};
                distanciaMenor = distancia;
            }
        });

        return mejor;
    },

    presionar(evento) {
        if (TL.modo !== "roi" || !TL.videoInfo) return;
        if (evento.button !== 0) return;

        const canvas = TL.$("roiCanvas");
        const punto = TL.puntoCanvas(evento, canvas);
        if (!punto) return;

        evento.preventDefault();
        canvas.setPointerCapture(evento.pointerId);

        this.terminarArrastre();

        const nodo = this.buscarNodo(punto);

        if (nodo?.tipo === "vertice") {
            this.seleccionado = nodo.indice;
            this.arrastrando = nodo.indice;
            this.origenArrastre = this.copia();
            this.movimiento = false;
        } else if (nodo?.tipo === "medio") {
            const antes = this.copia();
            const medio = this.puntosMedios().find(
                p => p.lado === nodo.indice
            );

            TL.roi.splice(nodo.indice + 1, 0, {
                x: Math.round(medio.x),
                y: Math.round(medio.y)
            });

            this.registrar(antes);
            this.seleccionado = nodo.indice + 1;
            this.arrastrando = this.seleccionado;
            this.origenArrastre = null;
            this.cambiado();
        } else {
            const antes = this.copia();

            TL.roi.push(punto);
            this.registrar(antes);

            this.seleccionado = TL.roi.length - 1;
            this.arrastrando = this.seleccionado;
            this.origenArrastre = null;
            this.cambiado();
        }

        TL.Video.dibujarOverlay();
    },

    mover(evento) {
        if (TL.modo !== "roi" || !TL.videoInfo) return;

        const canvas = TL.$("roiCanvas");
        const punto = TL.puntoCanvas(evento, canvas);
        if (!punto) return;

        if (this.arrastrando >= 0) {
            const anterior = TL.roi[this.arrastrando];

            if (
                anterior.x !== punto.x ||
                anterior.y !== punto.y
            ) {
                TL.roi[this.arrastrando] = punto;
                this.movimiento = true;
                this.cambiado();
            }
        } else {
            const nuevoHover = this.buscarNodo(punto);

            if (!this.iguales(this.hover, nuevoHover)) {
                this.hover = nuevoHover;
                TL.Video.dibujarOverlay();
            }

            canvas.style.cursor = nuevoHover
                ? "grab" : "crosshair";
        }
    },

    soltar(evento) {
        const canvas = TL.$("roiCanvas");

        this.terminarArrastre();

        if (canvas.hasPointerCapture(evento.pointerId)) {
            canvas.releasePointerCapture(evento.pointerId);
        }
    },

    terminarArrastre() {
        if (
            this.arrastrando >= 0 &&
            this.origenArrastre &&
            !this.iguales(this.origenArrastre, TL.roi)
        ) {
            this.registrar(this.origenArrastre);
        }

        this.arrastrando = -1;
        this.origenArrastre = null;
        this.movimiento = false;
    },

    deshacer() {
        this.terminarArrastre();

        if (!this.historial.length) {
            this.informar("No hay más acciones para deshacer.");
            return;
        }

        this.futuros.push(this.copia());
        TL.roi = this.historial.pop();
        this.seleccionado = -1;
        this.hover = null;
        this.cambiado();
    },

    rehacer() {
        this.terminarArrastre();

        if (!this.futuros.length) {
            this.informar("No hay acciones para rehacer.");
            return;
        }

        this.historial.push(this.copia());
        TL.roi = this.futuros.pop();
        this.seleccionado = -1;
        this.hover = null;
        this.cambiado();
    },

    eliminarSeleccionado() {
        this.terminarArrastre();

        if (
            this.seleccionado < 0 ||
            this.seleccionado >= TL.roi.length
        ) {
            this.informar("Selecciona primero un nodo amarillo.");
            return;
        }

        const antes = this.copia();

        TL.roi.splice(this.seleccionado, 1);
        this.registrar(antes);

        this.seleccionado = -1;
        this.hover = null;
        this.cambiado();
    },

    limpiar() {
        this.terminarArrastre();

        if (!TL.roi.length) return;

        const antes = this.copia();
        TL.roi = [];

        this.registrar(antes);
        this.seleccionado = -1;
        this.hover = null;
        this.cambiado();
    },

    area() {
        if (TL.roi.length < 3) return 0;

        let suma = 0;

        for (let i = 0; i < TL.roi.length; i++) {
            const a = TL.roi[i];
            const b = TL.roi[(i + 1) % TL.roi.length];
            suma += a.x * b.y - b.x * a.y;
        }

        return Math.abs(suma) / 2;
    },

    orientacion(a, b, c) {
        const valor =
            (b.x - a.x) * (c.y - a.y) -
            (b.y - a.y) * (c.x - a.x);

        return Math.sign(valor);
    },

    enSegmento(a, b, c) {
        return (
            c.x >= Math.min(a.x, b.x) &&
            c.x <= Math.max(a.x, b.x) &&
            c.y >= Math.min(a.y, b.y) &&
            c.y <= Math.max(a.y, b.y)
        );
    },

    segmentosCruzan(a, b, c, d) {
        const o1 = this.orientacion(a, b, c);
        const o2 = this.orientacion(a, b, d);
        const o3 = this.orientacion(c, d, a);
        const o4 = this.orientacion(c, d, b);

        if (o1 !== o2 && o3 !== o4) return true;

        return (
            (o1 === 0 && this.enSegmento(a, b, c)) ||
            (o2 === 0 && this.enSegmento(a, b, d)) ||
            (o3 === 0 && this.enSegmento(c, d, a)) ||
            (o4 === 0 && this.enSegmento(c, d, b))
        );
    },

    seCruza() {
        const puntos = TL.roi;
        const n = puntos.length;

        if (n < 4) return false;

        for (let i = 0; i < n; i++) {
            const a = puntos[i];
            const b = puntos[(i + 1) % n];

            for (let j = i + 1; j < n; j++) {
                if (
                    j === i ||
                    (j + 1) % n === i ||
                    (i + 1) % n === j
                ) continue;

                if (this.segmentosCruzan(
                    a, b,
                    puntos[j],
                    puntos[(j + 1) % n]
                )) {
                    return true;
                }
            }
        }

        return false;
    },

    alineados() {
        if (TL.roi.length < 3) return false;

        const a = TL.roi[0];
        const b = TL.roi.find(
            p => p.x !== a.x || p.y !== a.y
        );

        if (!b) return true;

        return TL.roi.every(
            p => this.orientacion(a, b, p) === 0
        );
    },

    validar() {
        if (TL.roi.length < 3) {
            return "Necesitas al menos tres puntos.";
        }

        if (this.alineados()) {
            return "Los puntos están alineados: área cero.";
        }

        if (this.seCruza()) {
            return "La ROI se cruza a sí misma.";
        }

        if (this.area() <= 0) {
            return "La ROI no tiene un área válida.";
        }

        return null;
    },

    informar(texto = null) {
        const error = TL.roi.length >= 3
            ? this.validar()
            : null;

        if (error) {
            TL.mensaje("mensajeRoi", error, "error");
            return;
        }

        TL.mensaje(
            "mensajeRoi",
            texto || (
                `${TL.roi.length} nodos · ` +
                `Área: ${this.area().toFixed(0)} px² · ` +
                (TL.roiGuardada ? "Guardada" : "Sin guardar")
            ),
            TL.roiGuardada ? "ok" : ""
        );
    },

    async guardar() {
        if (!TL.videoId) {
            this.informar("Primero carga un video.");
            return;
        }

        this.terminarArrastre();

        const error = this.validar();

        if (error) {
            TL.mensaje("mensajeRoi", error, "error");
            return;
        }

        try {
            const resultado = await TL.enviar(
                `/api/videos/${TL.videoId}/roi`,
                {puntos: TL.roi}
            );

            TL.roiGuardada = true;
            TL.calibracionGuardada = false;

            TL.mensaje(
                "mensajeRoi",
                `${resultado.mensaje} · ${TL.roi.length} nodos`,
                "ok"
            );
        } catch (error) {
            TL.mensaje("mensajeRoi", error.message, "error");
        }
    }
};
