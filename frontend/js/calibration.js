
TL.Calibration = {
    init() {
        TL.$("modoAnimal").addEventListener("click", () => {
            TL.modo = "animal";
            TL.mensaje(
                "posicionAnimal",
                "Selecciona el centro del animal en el video."
            );
            TL.$("video").scrollIntoView({behavior: "smooth"});
        });

        TL.$("quitarAnimal").addEventListener("click", () => {
            TL.animal = null;
            TL.calibracionGuardada = false;
            TL.Video.dibujarOverlay();

            TL.mensaje(
                "posicionAnimal",
                "Animal no seleccionado."
            );
        });

        TL.$("guardarCalibracion").addEventListener(
            "click", () => this.guardar()
        );

        TL.$("probarDeteccion").addEventListener(
            "click", () => this.probar()
        );

        const campos = [
            "diff", "open", "close", "mina",
            "maxa", "shrink", "sol"
        ];

        campos.forEach(id => {
            TL.$(id).addEventListener("input", () => {
                TL.calibracionGuardada = false;
                TL.mensaje(
                    "mensajeCalibracion",
                    "Hay cambios sin guardar."
                );
            });
        });
    },

    elegirAnimal(punto) {
        TL.animal = punto;
        TL.calibracionGuardada = false;

        TL.mensaje(
            "posicionAnimal",
            `Animal seleccionado: X=${punto.x}, Y=${punto.y}`,
            "ok"
        );

        TL.Video.dibujarOverlay();
    },

    parametros() {
        return {
            animal: TL.animal,
            diff: TL.numero("diff"),
            open_k: TL.numero("open"),
            close_k: TL.numero("close"),
            mina: TL.numero("mina"),
            maxa: TL.numero("maxa"),
            shrink: TL.numero("shrink"),
            sol: TL.numero("sol")
        };
    },

    async guardar() {
        if (!TL.videoId || !TL.roiGuardada || !TL.animal) {
            TL.mensaje(
                "mensajeCalibracion",
                "Guarda la ROI y selecciona el animal.",
                "error"
            );
            return;
        }

        try {
            const resultado = await TL.enviar(
                `/api/videos/${TL.videoId}/calibracion`,
                this.parametros()
            );

            TL.calibracionGuardada = true;

            TL.mensaje(
                "mensajeCalibracion",
                resultado.mensaje,
                "ok"
            );

        } catch (error) {
            TL.mensaje(
                "mensajeCalibracion",
                error.message,
                "error"
            );
        }
    },

    async probar() {
        if (!TL.videoId || !TL.calibracionGuardada) {
            TL.mensaje(
                "mensajeDeteccion",
                "Guarda primero la calibración.",
                "error"
            );
            return;
        }

        const boton = TL.$("probarDeteccion");
        boton.disabled = true;

        TL.mensaje("mensajeDeteccion", "Procesando detección...");

        try {
            const datos = await TL.api(
                `/api/videos/${TL.videoId}/deteccion/${TL.frame}`
            );

            TL.$("resultadoDeteccion").classList.remove("hidden");

            TL.$("imagenDeteccion").src =
                `data:image/jpeg;base64,${datos.imagen}`;

            TL.$("imagenMascara").src =
                `data:image/jpeg;base64,${datos.mascara}`;

            if (datos.encontrado) {
                TL.mensaje(
                    "mensajeDeteccion",
                    `Animal detectado en X=${datos.centro.x}, ` +
                    `Y=${datos.centro.y}`,
                    "ok"
                );
            } else {
                TL.mensaje(
                    "mensajeDeteccion",
                    "No se encontró el animal. Ajusta los parámetros.",
                    "error"
                );
            }

        } catch (error) {
            TL.mensaje(
                "mensajeDeteccion",
                error.message,
                "error"
            );
        } finally {
            boton.disabled = false;
        }
    }
};
