
TL.Heatmap = {
    canvas: null,
    controles: {},

    init() {
        const resultados = TL.$("resultados");

        const panel = TL.crearElemento(
            "div", {className: "chart-panel"}, resultados
        );

        TL.crearElemento(
            "h3", {text: "Mapa de calor del movimiento"}, panel
        );

        const descripcion = TL.crearElemento(
            "p", {className: "help"}, panel
        );

        descripcion.textContent =
            "Las zonas cálidas indican mayor concentración " +
            "de posiciones detectadas.";

        const controles = TL.crearElemento(
            "div", {className: "fields"}, panel
        );

        this.crearRango(
            controles, "radioCalor", "Radio", 5, 100, 30
        );

        this.crearRango(
            controles, "intensidadCalor", "Intensidad", 1, 10, 5
        );

        this.crearRango(
            controles, "opacidadCalor", "Opacidad", 10, 100, 75
        );

        const etiqueta = TL.crearElemento(
            "label", {text: "Fondo"}, controles
        );

        const selector = TL.crearElemento("select", {
            id: "fondoCalor"
        }, etiqueta);

        [
            ["oscuro", "Oscuro"],
            ["video", "Fotograma actual"],
            ["blanco", "Blanco"]
        ].forEach(([valor, texto]) => {
            const opcion = TL.crearElemento(
                "option", {value: valor, text: texto}, selector
            );
            opcion.value = valor;
        });

        selector.addEventListener(
            "change", () => this.actualizar()
        );

        this.canvas = TL.crearElemento("canvas", {
            id: "canvasHeatmap",
            width: "800",
            height: "450"
        }, panel);

        this.canvas.style.width = "100%";
        this.canvas.style.maxWidth = "800px";
        this.canvas.style.height = "auto";
        this.canvas.style.borderRadius = "8px";

        this.actualizar();
    },

    crearRango(padre, id, nombre, min, max, valor) {
        const etiqueta = TL.crearElemento(
            "label", {text: `${nombre}: ${valor}`}, padre
        );

        const control = TL.crearElemento("input", {
            id,
            type: "range",
            min: String(min),
            max: String(max),
            value: String(valor)
        }, etiqueta);

        control.addEventListener("input", () => {
            etiqueta.firstChild.textContent =
                `${nombre}: ${control.value}`;
            this.actualizar();
        });

        this.controles[id] = control;
    },

    color(t) {
        t = Math.max(0, Math.min(1, t));

        const paradas = [
            [0, [0, 30, 180]],
            [0.25, [0, 200, 255]],
            [0.5, [0, 230, 80]],
            [0.75, [255, 220, 0]],
            [1, [255, 30, 0]]
        ];

        for (let i = 1; i < paradas.length; i++) {
            if (t <= paradas[i][0]) {
                const [a, ca] = paradas[i - 1];
                const [b, cb] = paradas[i];
                const f = (t - a) / (b - a);

                return ca.map((v, j) =>
                    Math.round(v + (cb[j] - v) * f)
                );
            }
        }

        return [255, 30, 0];
    },

    actualizar() {
        if (!this.canvas) return;

        const canvas = this.canvas;
        const ctx = canvas.getContext("2d");
        const w = canvas.width;
        const h = canvas.height;

        const fondo = TL.$("fondoCalor")?.value || "oscuro";

        ctx.fillStyle = fondo === "blanco"
            ? "#ffffff" : "#152032";

        ctx.fillRect(0, 0, w, h);

        if (
            fondo === "video" &&
            TL.imagenFrame &&
            TL.videoInfo
        ) {
            const escala = Math.min(
                w / TL.videoInfo.ancho,
                h / TL.videoInfo.alto
            );

            const ancho = TL.videoInfo.ancho * escala;
            const alto = TL.videoInfo.alto * escala;

            ctx.drawImage(
                TL.imagenFrame,
                (w - ancho) / 2,
                (h - alto) / 2,
                ancho,
                alto
            );
        }

        if (!TL.videoInfo || !TL.trayectoria.length) return;

        // Resolución interna reducida para evitar procesar
        // millones de píxeles en cada actualización.
        const mapa = document.createElement("canvas");
        mapa.width = 200;
        mapa.height = 113;

        const mc = mapa.getContext("2d", {
            willReadFrequently: true
        });

        const escala = Math.min(
            mapa.width / TL.videoInfo.ancho,
            mapa.height / TL.videoInfo.alto
        );

        const dx =
            (mapa.width - TL.videoInfo.ancho * escala) / 2;

        const dy =
            (mapa.height - TL.videoInfo.alto * escala) / 2;

        const radio = Number(
            this.controles.radioCalor?.value || 30
        ) / 4;

        const intensidad = Number(
            this.controles.intensidadCalor?.value || 5
        );

        const opacidad = Number(
            this.controles.opacidadCalor?.value || 75
        ) / 100;

        mc.clearRect(0, 0, mapa.width, mapa.height);

        for (const [x, y] of TL.trayectoria) {
            const px = x * escala + dx;
            const py = y * escala + dy;

            const gradiente = mc.createRadialGradient(
                px, py, 0, px, py, radio
            );

            gradiente.addColorStop(0, "rgba(255,255,255,0.20)");
            gradiente.addColorStop(1, "rgba(255,255,255,0)");

            mc.fillStyle = gradiente;
            mc.fillRect(
                px - radio, py - radio,
                radio * 2, radio * 2
            );
        }

        const imagen = mc.getImageData(
            0, 0, mapa.width, mapa.height
        );

        const pixeles = imagen.data;

        let maximo = 0;

        for (let i = 3; i < pixeles.length; i += 4) {
            maximo = Math.max(maximo, pixeles[i]);
        }

        if (maximo === 0) return;

        for (let i = 0; i < pixeles.length; i += 4) {
            const nivel = Math.min(
                1,
                (pixeles[i + 3] / maximo) *
                (intensidad / 5)
            );

            const [r, g, b] = this.color(nivel);

            pixeles[i] = r;
            pixeles[i + 1] = g;
            pixeles[i + 2] = b;
            pixeles[i + 3] = Math.round(
                nivel * 255 * opacidad
            );
        }

        mc.putImageData(imagen, 0, 0);

        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(mapa, 0, 0, w, h);
    }
};
