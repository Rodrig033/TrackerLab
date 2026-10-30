import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk


_TK_ROOT = None


def set_tk_root(root):
    global _TK_ROOT
    _TK_ROOT = root


def get_tk_root():
    global _TK_ROOT

    if _TK_ROOT is not None:
        try:
            if _TK_ROOT.winfo_exists():
                return _TK_ROOT
        except Exception:
            _TK_ROOT = None

    _TK_ROOT = tk.Tk()
    _TK_ROOT.withdraw()

    return _TK_ROOT


def gui_info(title, msg):
    try:
        messagebox.showinfo(
            title,
            msg,
            parent=get_tk_root()
        )
    except Exception:
        print(msg)


def gui_warning(title, msg):
    try:
        messagebox.showwarning(
            title,
            msg,
            parent=get_tk_root()
        )
    except Exception:
        print(msg)


def gui_error(title, msg):
    try:
        messagebox.showerror(
            title,
            msg,
            parent=get_tk_root()
        )
    except Exception:
        print(msg)


def gui_ask_yes_no(
    title,
    msg,
    default_yes=True
):
    try:
        return bool(
            messagebox.askyesno(
                title,
                msg,
                parent=get_tk_root(),
                default=(
                    "yes"
                    if default_yes
                    else "no"
                )
            )
        )

    except Exception:
        return default_yes


def gui_ask_string(
    title,
    prompt,
    default=""
):
    try:
        result = simpledialog.askstring(
            title,
            prompt,
            initialvalue=default,
            parent=get_tk_root()
        )

        if result is None:
            return None

        return str(
            result
        ).strip()

    except Exception:
        return default


def gui_ask_float(
    title,
    prompt,
    default=None
):
    while True:
        value = gui_ask_string(
            title,
            prompt,
            (
                ""
                if default is None
                else str(default)
            )
        )

        if value is None:
            return None

        if value == "":
            return default

        try:
            return float(
                value
            )

        except ValueError:
            gui_warning(
                "Valor inválido",
                "Escribe un número válido."
            )


def ask_yes_no(
    prompt,
    default_yes=True
):
    return gui_ask_yes_no(
        "Confirmar",
        prompt,
        default_yes=default_yes
    )


def select_video_file():
    path = filedialog.askopenfilename(
        parent=get_tk_root(),
        title="Seleccionar video",
        filetypes=[
            (
                "Archivos de video",
                "*.mp4 *.avi *.mov *.mkv *.wmv *.mpeg *.mpg *.m4v"
            ),
            (
                "Todos los archivos",
                "*.*"
            )
        ]
    )

    if not path:
        return None

    return os.path.abspath(
        path
    )


def gui_select_reuse_options(
    exp_name
):
    parent = get_tk_root()

    window = tk.Toplevel(
        parent
    )

    window.title(
        "Reutilizar calibración"
    )

    window.geometry(
        "650x560"
    )

    window.minsize(
        580,
        500
    )

    window.resizable(
        True,
        True
    )

    try:
        window.transient(
            parent
        )
    except Exception:
        pass

    try:
        window.grab_set()
    except Exception:
        pass

    result = {
        "accepted": False
    }

    use_roi = tk.BooleanVar(
        value=True
    )

    use_zones = tk.BooleanVar(
        value=True
    )

    use_scale = tk.BooleanVar(
        value=True
    )

    use_origin = tk.BooleanVar(
        value=True
    )

    use_detection = tk.BooleanVar(
        value=True
    )

    use_mode = tk.BooleanVar(
        value=True
    )

    container = ttk.Frame(
        window,
        padding=22
    )

    container.pack(
        fill="both",
        expand=True
    )

    ttk.Label(
        container,
        text="Calibración encontrada",
        font=(
            "Arial",
            16,
            "bold"
        )
    ).pack(
        anchor="w"
    )

    ttk.Label(
        container,
        text=f"Experimento: {exp_name}",
        font=(
            "Arial",
            10
        )
    ).pack(
        anchor="w",
        pady=(6, 4)
    )

    ttk.Label(
        container,
        text=(
            "Selecciona qué partes de la calibración "
            "anterior deseas reutilizar."
        ),
        wraplength=560
    ).pack(
        anchor="w",
        pady=(4, 16)
    )

    options = ttk.LabelFrame(
        container,
        text="Configuración disponible",
        padding=15
    )

    options.pack(
        fill="both",
        expand=True
    )

    ttk.Checkbutton(
        options,
        text="Región de interés (ROI)",
        variable=use_roi
    ).pack(
        anchor="w",
        pady=5
    )

    ttk.Label(
        options,
        text="Área general utilizada para realizar el análisis."
    ).pack(
        anchor="w",
        padx=(25, 0)
    )

    ttk.Checkbutton(
        options,
        text="Zonas internas",
        variable=use_zones
    ).pack(
        anchor="w",
        pady=(12, 5)
    )

    ttk.Label(
        options,
        text="Regiones definidas dentro del área experimental."
    ).pack(
        anchor="w",
        padx=(25, 0)
    )

    ttk.Checkbutton(
        options,
        text="Escala espacial",
        variable=use_scale
    ).pack(
        anchor="w",
        pady=(12, 5)
    )

    ttk.Checkbutton(
        options,
        text="Origen del sistema XY",
        variable=use_origin
    ).pack(
        anchor="w",
        pady=5
    )

    ttk.Checkbutton(
        options,
        text="Parámetros de detección",
        variable=use_detection
    ).pack(
        anchor="w",
        pady=5
    )

    ttk.Checkbutton(
        options,
        text="Modo de seguimiento",
        variable=use_mode
    ).pack(
        anchor="w",
        pady=5
    )

    buttons = ttk.Frame(
        container
    )

    buttons.pack(
        fill="x",
        pady=(18, 0)
    )

    def accept():
        result["accepted"] = True
        window.destroy()

    def new_calibration():
        use_roi.set(False)
        use_zones.set(False)
        use_scale.set(False)
        use_origin.set(False)
        use_detection.set(False)
        use_mode.set(False)

        result["accepted"] = True

        window.destroy()

    def cancel():
        result["accepted"] = False
        window.destroy()

    ttk.Button(
        buttons,
        text="Cancelar",
        command=cancel
    ).pack(
        side="right",
        padx=(8, 0)
    )

    ttk.Button(
        buttons,
        text="Nueva calibración",
        command=new_calibration
    ).pack(
        side="right",
        padx=(8, 0)
    )

    ttk.Button(
        buttons,
        text="Reutilizar",
        command=accept
    ).pack(
        side="right"
    )

    window.protocol(
        "WM_DELETE_WINDOW",
        cancel
    )

    try:
        window.wait_window()
    except Exception:
        pass

    if not result[
        "accepted"
    ]:
        return None

    return {
        "roi": bool(
            use_roi.get()
        ),

        "zones": bool(
            use_zones.get()
        ),

        "scale": bool(
            use_scale.get()
        ),

        "origin": bool(
            use_origin.get()
        ),

        "detection": bool(
            use_detection.get()
        ),

        "mode": bool(
            use_mode.get()
        )
    }