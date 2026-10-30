
TL.Tracking = {
    init() {
        TL.$("iniciarTracking").addEventListener(
            "click", () => this.iniciar()
        );

        TL.$("detenerTracking").addEventListener(
            "click", () => this.detener()
        );

        TL.$("descargarCSV").addEventListener("click", () => {
            if (!TL.trabajoId) return;

            window.location.href =
                `/api/seguimientos/${TL.trabajoId}/csv`;
        });

        this.reiniciar();
    },

    reiniciar() {
        this.finalizar();

        TL.trayectoria = [];
        TL.series = [];
        TL.metricas = {};

        [
            "framesProcesados",
            "framesDetectados",
            "framesRechazados",
            "framesRecuperados"
        ].forEach(id => {
            TL.$(id).textContent = "0";
        });

        [
            "tiempoAnalizado",
            "porcentajeDeteccion",
            "distanciaTotal",
            "velocidadPromedio",
            "aceleracionPromedio",
            "anguloPromedio",
            "velocidadAngular",
            "tiempoValido"
        ].forEach(id => {
            TL.$(id).textContent = "—";
        });

        TL.$("progresoTracking").value = 0;
        TL.$("porcentajeTracking").textContent = "0%";
        TL.$("estadoTracking").textContent = "Sin iniciar";
        TL.$("descargarCSV").disabled = true;

        TL.mensaje("mensajeResultados", "");

        TL.Charts.actualizar();
        TL.Heatmap.actualizar();
        TL.Zones.actualizarEstadisticas();
    },

    async iniciar() {
        if (!TL.videoId || !TL.calibracionGuardada) {
            TL.mensaje(
                "mensajeResultados",
                "Carga un video, guarda la ROI y calibra el animal.",
                "error"
            );
            return;
        }

        if (TL.intervalo) return;

        const inicio = TL.numero("inicioTracking");
        const fin = TL.numero("finTracking");
        const salto = TL.numero("saltoTracking");

        if (
            !Number.isInteger(inicio) ||
            !Number.isInteger(fin) ||
            !Number.isInteger(salto) ||
            inicio < 0 ||
            fin < inicio ||
            fin >= TL.videoInfo.frames ||
            salto < 1
        ) {
            TL.mensaje(
                "mensajeResultados",
                "Revisa el intervalo y el salto de fotogramas.",
                "error"
            );
            return;
        }

        if (TL.frame !== inicio) {
            TL.mensaje(
                "mensajeResultados",
                "Aviso: la posición inicial del animal debe " +
                "corresponder al fotograma inicial del seguimiento."
            );
        }

        this.reiniciar();

        TL.$("iniciarTracking").disabled = true;
        TL.$("estadoTracking").textContent = "Preparando...";

        try {
            const resultado = await TL.enviar(
                `/api/videos/${TL.videoId}/seguimiento`,
                {inicio, fin, salto}
            );

            TL.trabajoId = resultado.trabajo_id;

            TL.$("detenerTracking").disabled = false;

            TL.intervalo = setInterval(
                () => this.consultar(),
                1000
            );

            await this.consultar();

        } catch (error) {
            this.finalizar();

            TL.$("estadoTracking").textContent = "Error";
            TL.mensaje(
                "mensajeResultados",
                error.message,
                "error"
            );
        }
    },

    async consultar() {
        if (!TL.trabajoId || TL.consultando) return;

        TL.consultando = true;

        try {
            const datos = await TL.api(
                `/api/seguimientos/${TL.trabajoId}`
            );

            TL.$("estadoTracking").textContent =
                `Estado: ${datos.estado}`;

            TL.$("progresoTracking").value =
                datos.porcentaje || 0;

            TL.$("porcentajeTracking").textContent =
                `${datos.porcentaje || 0}%`;

            TL.$("framesProcesados").textContent =
                datos.procesados || 0;

            TL.$("framesDetectados").textContent =
                datos.detectados || 0;

            TL.$("framesRechazados").textContent =
                datos.rechazados || 0;

            TL.$("framesRecuperados").textContent =
                datos.recuperados || 0;

            TL.trayectoria = datos.trayectoria || [];
            TL.series = datos.series || [];
            TL.metricas = datos.metricas || {};

            this.mostrarMetricas(TL.metricas);

            TL.Charts.actualizar();
            TL.Heatmap.actualizar();
            TL.Zones.actualizarEstadisticas();

            if (datos.estado === "error") {
                TL.mensaje(
                    "mensajeResultados",
                    datos.error || "Error desconocido.",
                    "error"
                );
                this.finalizar();
            }

            if (
                datos.estado === "finalizado" ||
                datos.estado === "detenido"
            ) {
                this.finalizar();

                TL.$("descargarCSV").disabled = false;

                const avisos = datos.advertencias || [];

                TL.mensaje(
                    "mensajeResultados",
                    "Seguimiento terminado. " +
                    (avisos.length ? avisos.join(" | ") : ""),
                    "ok"
                );
            }

        } catch (error) {
            TL.mensaje(
                "mensajeResultados",
                error.message,
                "error"
            );

            this.finalizar();

        } finally {
            TL.consultando = false;
        }
    },

    mostrarMetricas(m) {
        const valores = {
            tiempoAnalizado: [m.tiempo_s, "s"],
            porcentajeDeteccion: [m.deteccion_pct, "%"],
            distanciaTotal: [m.distancia_px, "px"],
            velocidadPromedio: [m.velocidad_px_s, "px/s"],
            aceleracionPromedio: [m.aceleracion_px_s2, "px/s²"],
            anguloPromedio: [m.angulo_deg, "°"],
            velocidadAngular: [m.velocidad_angular_deg_s, "°/s"],
            tiempoValido: [m.tiempo_valido_s, "s"]
        };

        Object.entries(valores).forEach(([id, [valor, unidad]]) => {
            TL.$(id).textContent = TL.formato(valor, unidad);
        });
    },

    async detener() {
        if (!TL.trabajoId) return;

        try {
            await TL.api(
                `/api/seguimientos/${TL.trabajoId}/detener`,
                {method: "POST"}
            );

            TL.$("estadoTracking").textContent = "Deteniendo...";

        } catch (error) {
            TL.mensaje(
                "mensajeResultados",
                error.message,
                "error"
            );
        }
    },

    finalizar() {
        if (TL.intervalo) {
            clearInterval(TL.intervalo);
            TL.intervalo = null;
        }

        TL.$("iniciarTracking").disabled = false;
        TL.$("detenerTracking").disabled = true;
    }
};
