
TL.Exports = {
    panel: null,
    lista: null,
    estado: null,
    boton: null,
    ultimoTrabajo: null,

    init() {
        const seccion = TL.$("resultados");
        if (!seccion) return;

        this.panel = document.createElement("div");
        this.panel.className = "chart-panel";
        this.panel.id = "archivos-generados";

        this.panel.innerHTML = `
            <h3>Archivos generados</h3>
            <p class="help">
                Descarga las gráficas, CSV, Excel, PDF
                o todos los resultados en ZIP.
                No se incluye dataset.
            </p>
            <div class="button-row">
                <button
                    type="button"
                    class="btn primary"
                    id="generarArchivos">
                    Generar archivos
                </button>
            </div>
            <p class="message" id="estadoArchivos"></p>
            <div id="listaArchivos" class="files-list"></div>
        `;

        seccion.appendChild(this.panel);

        this.lista = this.panel.querySelector("#listaArchivos");
        this.estado = this.panel.querySelector("#estadoArchivos");
        this.boton = this.panel.querySelector("#generarArchivos");

        this.boton.addEventListener("click", () => {
            this.generar();
        });
    },

    async generar() {
        if (!TL.trabajoId) {
            this.estado.textContent =
                "Primero ejecuta un seguimiento.";
            return;
        }

        this.boton.disabled = true;
        this.estado.textContent =
            "Generando archivos, espera un momento...";

        try {
            const respuesta = await TL.api(
                `/api/exportaciones/${encodeURIComponent(TL.trabajoId)}/generar`,
                {method: "POST"}
            );

            this.ultimoTrabajo = TL.trabajoId;

            this.renderizar(respuesta.archivos || []);

            this.estado.textContent =
                "Archivos generados correctamente.";

        } catch (error) {
            this.estado.textContent = error.message;
        } finally {
            this.boton.disabled = false;
        }
    },

    async actualizar() {
        if (!TL.trabajoId) return;

        try {
            const respuesta = await TL.api(
                `/api/exportaciones/${encodeURIComponent(TL.trabajoId)}`
            );

            this.ultimoTrabajo = TL.trabajoId;

            this.renderizar(respuesta.archivos || []);

        } catch (error) {
            this.estado.textContent = error.message;
        }
    },

    renderizar(archivos) {
        this.lista.replaceChildren();

        if (!archivos.length) {
            this.estado.textContent =
                "Todavía no hay archivos exportados.";
            return;
        }

        for (const nombre of archivos) {
            const fila = document.createElement("div");
            fila.className = "button-row";
            fila.style.marginBottom = "8px";

            const etiqueta = document.createElement("span");
            etiqueta.textContent = nombre;
            etiqueta.style.flex = "1";

            const enlace = document.createElement("a");
            enlace.className = "btn";

            enlace.textContent = nombre.endsWith(".zip")
                ? "Descargar todo"
                : "Descargar";

            enlace.href =
                `/api/exportaciones/${encodeURIComponent(this.ultimoTrabajo)}` +
                `/archivo/${encodeURIComponent(nombre)}`;

            fila.append(etiqueta, enlace);
            this.lista.appendChild(fila);
        }
    }
};

document.addEventListener("DOMContentLoaded", () => {
    if (window.TL && TL.Exports && !TL.Exports.panel) {
        TL.Exports.init();
    }
});
