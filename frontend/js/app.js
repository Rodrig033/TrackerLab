
window.TL = {
    videoId: null,
    videoInfo: null,
    frame: 0,
    roi: [],
    roiGuardada: false,
    animal: null,
    calibracionGuardada: false,
    modo: "ninguno",
    trabajoId: null,
    intervalo: null,
    consultando: false,
    trayectoria: [],
    series: [],
    zonas: [],
    metricas: {},
    solicitudFrame: 0,
    imagenFrame: null,

    $: id => document.getElementById(id),

    async api(url, opciones = {}) {
        let respuesta;

        try {
            respuesta = await fetch(url, opciones);
        } catch {
            throw new Error("No se pudo conectar con el servidor.");
        }

        if (!respuesta.ok) {
            let detalle = `Error HTTP ${respuesta.status}`;

            try {
                const datos = await respuesta.json();
                detalle = typeof datos.detail === "string"
                    ? datos.detail
                    : JSON.stringify(datos.detail || datos);
            } catch {}

            throw new Error(detalle);
        }

        const tipo = respuesta.headers.get("content-type") || "";
        return tipo.includes("application/json")
            ? respuesta.json()
            : respuesta;
    },

    async enviar(url, datos) {
        return this.api(url, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(datos)
        });
    },

    mensaje(id, texto, tipo = "") {
        const elemento = this.$(id);
        if (!elemento) return;

        elemento.textContent = texto;
        elemento.classList.remove("ok", "error");
        if (tipo) elemento.classList.add(tipo);
    },

    numero(id, defecto = 0) {
        const valor = Number(this.$(id)?.value);
        return Number.isFinite(valor) ? valor : defecto;
    },

    formato(valor, unidad = "") {
        return valor == null || !Number.isFinite(Number(valor))
            ? "—"
            : `${Number(valor).toFixed(2)} ${unidad}`.trim();
    },

    crearElemento(etiqueta, atributos = {}, padre = null) {
        const elemento = document.createElement(etiqueta);

        Object.entries(atributos).forEach(([clave, valor]) => {
            if (clave === "text") elemento.textContent = valor;
            else if (clave === "className") elemento.className = valor;
            else elemento.setAttribute(clave, valor);
        });

        if (padre) padre.appendChild(elemento);
        return elemento;
    },

    puntoCanvas(evento, canvas) {
        const rect = canvas.getBoundingClientRect();

        if (!rect.width || !rect.height) return null;

        return {
            x: Math.max(0, Math.min(
                canvas.width - 1,
                Math.round(
                    (evento.clientX - rect.left) *
                    canvas.width / rect.width
                )
            )),
            y: Math.max(0, Math.min(
                canvas.height - 1,
                Math.round(
                    (evento.clientY - rect.top) *
                    canvas.height / rect.height
                )
            ))
        };
    },

    async comprobarServidor() {
        try {
            await this.api("/api/estado");
            this.$("indicadorServidor").className = "status-dot online";
            this.$("textoServidor").textContent = "Servidor conectado";
        } catch {
            this.$("indicadorServidor").className = "status-dot offline";
            this.$("textoServidor").textContent = "Servidor desconectado";
        }
    },

    iniciar() {
        const modulos = [
            "Video", "ROI", "Calibration", "Tracking",
            "Charts", "Heatmap", "Zones"
        ];

        for (const nombre of modulos) {
            const modulo = this[nombre];

            if (!modulo || typeof modulo.init !== "function") {
                console.error(`Falta el módulo ${nombre}`);
                continue;
            }

            modulo.init();
        }

        document.querySelectorAll(".nav-link").forEach(enlace => {
            enlace.addEventListener("click", () => {
                document.querySelectorAll(".nav-link").forEach(e =>
                    e.classList.remove("active")
                );
                enlace.classList.add("active");
            });
        });

        this.comprobarServidor();
    }
};

document.addEventListener("DOMContentLoaded", () => TL.iniciar());
