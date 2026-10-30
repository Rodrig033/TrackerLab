
TL.Video = {
    init() {
        TL.$("subirVideo").addEventListener(
            "click", () => this.cargar()
        );

        TL.$("slider").addEventListener("input", () => {
            this.mostrarFrame(TL.numero("slider"));
        });

        TL.$("roiCanvas").addEventListener("click", evento => {
            if (TL.modo === "roi") return;

            const punto = TL.puntoCanvas(
                evento, TL.$("roiCanvas")
            );

            if (!punto) return;

            if (TL.modo === "animal") {
                TL.Calibration.elegirAnimal(punto);
            }

            // El modo zona es gestionado por zones.js.
        });

        window.addEventListener("resize", () => {
            this.dibujarOverlay();
        });
    },

    async cargar() {
        const archivo = TL.$("archivo").files[0];

        if (!archivo) {
            TL.mensaje(
                "mensajeVideo",
                "Selecciona un video.",
                "error"
            );
            return;
        }

        if (TL.intervalo) {
            TL.mensaje(
                "mensajeVideo",
                "Espera a que termine el seguimiento.",
                "error"
            );
            return;
        }

        const formulario = new FormData();
        formulario.append("archivo", archivo);

        TL.$("subirVideo").disabled = true;
        TL.mensaje("mensajeVideo", "Cargando video...");

        try {
            const info = await TL.api("/api/videos", {
                method: "POST",
                body: formulario
            });

            TL.videoId = info.id;
            TL.videoInfo = info;
            TL.frame = 0;
            TL.roi = [];
            TL.animal = null;
            TL.roiGuardada = false;
            TL.calibracionGuardada = false;
            TL.trayectoria = [];
            TL.series = [];
            TL.zonas = [];
            TL.trabajoId = null;
            TL.metricas = {};
            TL.modo = "ninguno";

            TL.ROI.reiniciarEditor();

            TL.$("slider").max = info.frames - 1;
            TL.$("slider").value = 0;

            TL.$("inicioTracking").value = 0;
            TL.$("finTracking").value = info.frames - 1;

            const canvas = TL.$("roiCanvas");
            canvas.width = info.ancho;
            canvas.height = info.alto;

            canvas.style.touchAction = "none";

            TL.$("contenedorVideo").classList.add("visible");
            TL.$("videoPlaceholder").classList.add("hidden");

            TL.$("informacionVideo").textContent =
                `${info.nombre} · ${info.ancho}×${info.alto} · ` +
                `${info.frames} frames · ` +
                `${info.fps.toFixed(2)} FPS`;

            TL.mensaje(
                "mensajeVideo",
                "Video cargado correctamente.",
                "ok"
            );

            TL.mensaje("mensajeRoi", "");
            TL.mensaje(
                "posicionAnimal",
                "Animal no seleccionado."
            );
            TL.mensaje("mensajeCalibracion", "");
            TL.mensaje("mensajeDeteccion", "");

            TL.$("resultadoDeteccion").classList.add("hidden");

            TL.Tracking.reiniciar();
            TL.Zones.limpiar(false);
            TL.Heatmap.actualizar();

            this.mostrarFrame(0);

        } catch (error) {
            TL.mensaje(
                "mensajeVideo",
                error.message,
                "error"
            );
        } finally {
            TL.$("subirVideo").disabled = false;
        }
    },

    mostrarFrame(numero) {
        if (!TL.videoId) return;

        numero = Math.max(
            0,
            Math.min(
                TL.videoInfo.frames - 1,
                Number(numero)
            )
        );

        TL.frame = numero;
        TL.$("slider").value = numero;
        TL.$("numeroFrame").textContent = numero;

        const solicitud = ++TL.solicitudFrame;
        const imagen = new Image();

        imagen.onload = () => {
            if (solicitud !== TL.solicitudFrame) return;

            TL.imagenFrame = imagen;
            TL.$("imagen").src = imagen.src;

            this.dibujarOverlay();
            TL.Heatmap.actualizar();
        };

        imagen.onerror = () => {
            if (solicitud !== TL.solicitudFrame) return;

            TL.mensaje(
                "mensajeVideo",
                `No se pudo cargar el frame ${numero}.`,
                "error"
            );
        };

        imagen.src =
            `/api/videos/${TL.videoId}/frame/${numero}`;
    },

    dibujarOverlay() {
        const canvas = TL.$("roiCanvas");
        const ctx = canvas.getContext("2d");

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (!TL.videoInfo) return;

        this.dibujarROI(ctx);
        this.dibujarAnimal(ctx);

        if (TL.Zones?.dibujarSobreVideo) {
            TL.Zones.dibujarSobreVideo(ctx);
        }
    },

    dibujarROI(ctx) {
        const puntos = TL.roi;
        if (!puntos.length) return;

        const escala = Math.max(
            TL.ROI.escala(), 0.000001
        );

        const grosor = 3 / escala;
        const radioVertice = 7 / escala;
        const radioMedio = 6 / escala;

        ctx.save();

        ctx.lineWidth = grosor;
        ctx.strokeStyle = "#00ee38";
        ctx.fillStyle = "rgba(0, 220, 60, 0.08)";

        ctx.beginPath();
        ctx.moveTo(puntos[0].x, puntos[0].y);

        for (let i = 1; i < puntos.length; i++) {
            ctx.lineTo(puntos[i].x, puntos[i].y);
        }

        if (puntos.length >= 3) {
            ctx.closePath();
            ctx.fill();
        }

        ctx.stroke();

        // Nodos blancos: puntos medios de cada lado.
        if (puntos.length >= 3 && TL.modo === "roi") {
            TL.ROI.puntosMedios().forEach(medio => {
                const activo =
                    TL.ROI.hover?.tipo === "medio" &&
                    TL.ROI.hover.indice === medio.lado;

                ctx.beginPath();
                ctx.arc(
                    medio.x,
                    medio.y,
                    activo ? radioMedio * 1.35 : radioMedio,
                    0,
                    Math.PI * 2
                );

                ctx.fillStyle = activo
                    ? "#38bdf8" : "#ffffff";

                ctx.fill();
                ctx.lineWidth = 2 / escala;
                ctx.strokeStyle = "#111827";
                ctx.stroke();
            });
        }

        // Nodos amarillos: vértices arrastrables.
        puntos.forEach((punto, indice) => {
            const seleccionado =
                TL.ROI.seleccionado === indice;

            const hover =
                TL.ROI.hover?.tipo === "vertice" &&
                TL.ROI.hover.indice === indice;

            const radio = seleccionado || hover
                ? radioVertice * 1.3
                : radioVertice;

            ctx.beginPath();
            ctx.arc(
                punto.x,
                punto.y,
                radio,
                0,
                Math.PI * 2
            );

            ctx.fillStyle = seleccionado
                ? "#f97316"
                : "#facc15";

            ctx.fill();

            if (seleccionado || hover) {
                ctx.lineWidth = 2 / escala;
                ctx.strokeStyle = "#ffffff";
                ctx.stroke();
            }
        });

        ctx.restore();
    },

    dibujarAnimal(ctx) {
        if (!TL.animal) return;

        const {x, y} = TL.animal;
        const escala = Math.max(
            TL.ROI.escala(), 0.000001
        );

        const radio = 11 / escala;

        ctx.save();
        ctx.strokeStyle = "#ef4444";
        ctx.fillStyle = "#ef4444";
        ctx.lineWidth = 3 / escala;

        ctx.beginPath();
        ctx.arc(x, y, radio, 0, Math.PI * 2);
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(x, y, 4 / escala, 0, Math.PI * 2);
        ctx.fill();

        ctx.beginPath();
        ctx.moveTo(x - radio * 1.5, y);
        ctx.lineTo(x + radio * 1.5, y);
        ctx.moveTo(x, y - radio * 1.5);
        ctx.lineTo(x, y + radio * 1.5);
        ctx.stroke();

        ctx.restore();
    }
};
