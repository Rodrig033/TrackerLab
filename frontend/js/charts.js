
TL.Charts = {
    init() {
        const panel = TL.$("canvasTrayectoria").parentElement;

        const controles = TL.crearElemento(
            "div", {className: "button-row"}, panel
        );

        TL.crearElemento(
            "label",
            {text: "Grosor de trayectoria: "},
            controles
        );

        const grosor = TL.crearElemento("input", {
            id: "grosorTrayectoria",
            type: "range",
            min: "1",
            max: "10",
            value: "2"
        }, controles);

        grosor.addEventListener("input", () => this.actualizar());

        this.actualizar();
    },

    limpiar(id, fondo = "#ffffff") {
        const canvas = TL.$(id);
        const ctx = canvas.getContext("2d");

        ctx.fillStyle = fondo;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        return {canvas, ctx};
    },

    actualizar() {
        this.trayectoria();
        this.lineas(
            "graficaVelocidad", "velocidad", "#2563eb", "px/s"
        );
        this.lineas(
            "graficaDistancia", "distancia", "#16a34a", "px"
        );
    },

    trayectoria() {
        const {canvas, ctx} = this.limpiar(
            "canvasTrayectoria", "#152032"
        );

        if (!TL.videoInfo) return;

        const escala = Math.min(
            canvas.width / TL.videoInfo.ancho,
            canvas.height / TL.videoInfo.alto
        );

        const dx =
            (canvas.width - TL.videoInfo.ancho * escala) / 2;

        const dy =
            (canvas.height - TL.videoInfo.alto * escala) / 2;

        TL.Zones.dibujarEnTrayectoria(ctx, escala, dx, dy);

        const puntos = TL.trayectoria;
        if (!puntos.length) return;

        ctx.strokeStyle = "#34d399";
        ctx.lineWidth = Number(
            TL.$("grosorTrayectoria")?.value || 2
        );

        ctx.beginPath();

        puntos.forEach(([x, y], i) => {
            const px = x * escala + dx;
            const py = y * escala + dy;

            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        });

        ctx.stroke();

        const marcar = (p, color) => {
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(
                p[0] * escala + dx,
                p[1] * escala + dy,
                5, 0, Math.PI * 2
            );
            ctx.fill();
        };

        marcar(puntos[0], "#facc15");
        marcar(puntos[puntos.length - 1], "#ef4444");
    },

    lineas(id, propiedad, color, unidad) {
        const {canvas, ctx} = this.limpiar(id);

        const puntos = TL.series.filter(p =>
            Number.isFinite(p.tiempo) &&
            Number.isFinite(p[propiedad])
        );

        if (puntos.length < 2) return;

        const margen = 55;
        const ancho = canvas.width - margen * 2;
        const alto = canvas.height - margen * 2;

        const tiempos = puntos.map(p => p.tiempo);
        const valores = puntos.map(p => p[propiedad]);

        const minX = Math.min(...tiempos);
        const maxX = Math.max(...tiempos);
        const minY = Math.min(0, ...valores);
        const maxY = Math.max(1, ...valores);

        const rangoX = Math.max(0.001, maxX - minX);
        const rangoY = Math.max(0.001, maxY - minY);

        ctx.strokeStyle = "#cbd5e1";
        ctx.lineWidth = 1;

        ctx.beginPath();
        ctx.moveTo(margen, margen);
        ctx.lineTo(margen, canvas.height - margen);
        ctx.lineTo(canvas.width - margen, canvas.height - margen);
        ctx.stroke();

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();

        puntos.forEach((p, i) => {
            const x = margen +
                (p.tiempo - minX) / rangoX * ancho;

            const y = canvas.height - margen -
                (p[propiedad] - minY) / rangoY * alto;

            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });

        ctx.stroke();

        ctx.fillStyle = "#475569";
        ctx.font = "13px Arial";

        ctx.fillText(unidad, 12, 20);
        ctx.fillText(maxY.toFixed(1), 8, margen);
        ctx.fillText(minY.toFixed(1), 8, canvas.height - margen);
        ctx.fillText(
            `${minX.toFixed(1)} s`,
            margen,
            canvas.height - 15
        );
        ctx.fillText(
            `${maxX.toFixed(1)} s`,
            canvas.width - margen - 65,
            canvas.height - 15
        );
    }
};
