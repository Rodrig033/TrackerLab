import os
import sys
import tkinter as tk

import cv2

from tkinter import filedialog, messagebox, ttk

from ui.dialogs import set_tk_root


CSV_BASE_NAME = "track_RATA"


def setup_windows_support():
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "TrackerLab.2026"
        )

    except Exception:
        pass


def project_directory():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)

    return os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )


def resource_path(filename):
    if getattr(sys, "frozen", False):
        try:
            base = sys._MEIPASS
        except Exception:
            base = os.path.dirname(sys.executable)
    else:
        base = project_directory()

    return os.path.join(
        base,
        "resources",
        filename
    )


def set_app_icon(window):
    try:
        ico_path = resource_path(
            "Tracker_Lab_icon.ico"
        )

        if os.path.exists(ico_path):
            window.iconbitmap(ico_path)

    except Exception:
        pass

    try:
        png_path = resource_path(
            "Tracker_Lab_logo.png"
        )

        if os.path.exists(png_path):
            icon_img = tk.PhotoImage(
                file=png_path
            )

            window.iconphoto(
                True,
                icon_img
            )

            window._tracker_lab_icon = icon_img

    except Exception:
        pass


def show_splash_screen(duration_ms=None):
    splash = tk.Tk()

    splash.title("TrackerLab")

    set_app_icon(splash)

    splash.configure(
        bg="white"
    )

    splash.resizable(
        False,
        False
    )

    width = 760
    height = 690

    screen_w = splash.winfo_screenwidth()
    screen_h = splash.winfo_screenheight()

    x = max(
        0,
        (screen_w // 2) - (width // 2)
    )

    y = max(
        0,
        (screen_h // 2) - (height // 2)
    )

    splash.geometry(
        f"{width}x{height}+{x}+{y}"
    )

    try:
        splash.attributes(
            "-topmost",
            True
        )
    except Exception:
        pass

    frame = tk.Frame(
        splash,
        bg="white",
        padx=28,
        pady=20
    )

    frame.pack(
        fill="both",
        expand=True
    )

    logo_path = resource_path(
        "Tracker_Lab_logo.png"
    )

    try:
        logo_img = tk.PhotoImage(
            file=logo_path
        )

        max_side = 210

        factor = max(
            1,
            int(
                max(
                    logo_img.width(),
                    logo_img.height()
                ) / max_side
            )
        )

        if factor > 1:
            logo_img = logo_img.subsample(
                factor,
                factor
            )

        logo_label = tk.Label(
            frame,
            image=logo_img,
            bg="white"
        )

        logo_label.image = logo_img

        logo_label.pack(
            pady=(0, 10)
        )

    except Exception:
        pass

    title = tk.Label(
        frame,
        text="TrackerLab",
        bg="white",
        fg="black",
        font=(
            "Segoe UI",
            20,
            "bold"
        )
    )

    title.pack(
        pady=(0, 12)
    )

    body_text = (
        "Este software fue desarrollado por José Abraham Rivera Uribe "
        "(Universidad Veracruzana), con asistencia de inteligencia artificial "
        "y bajo la dirección de Alejandro León, Laboratorio de Psicología "
        "Comparada, Universidad Veracruzana.\n\n"

        "Xalapa, Veracruz, México · 01 de mayo de 2026\n\n"

        "This software was developed by José Abraham Rivera Uribe "
        "(Universidad Veracruzana), with the assistance of artificial "
        "intelligence and under the supervision of Alejandro León, "
        "Laboratory of Comparative Psychology, Universidad Veracruzana.\n\n"

        "Xalapa, Veracruz, Mexico · May 1, 2026\n\n"

        "Para fines académicos, se solicita su adecuada citación / "
        "For academic purposes, proper citation is requested:\n\n"

        "Rivera, A., & León, A. (2026). TrackerLab "
        "[Software en desarrollo / Software in development]. "
        "Laboratorio de Psicología Comparada, Universidad Veracruzana."
    )

    body = tk.Label(
        frame,
        text=body_text,
        bg="white",
        fg="black",
        font=(
            "Segoe UI",
            10
        ),
        justify="center",
        wraplength=690
    )

    body.pack(
        pady=(0, 14)
    )

    btn_frame = tk.Frame(
        frame,
        bg="white"
    )

    btn_frame.pack(
        side="bottom",
        fill="x",
        pady=(8, 0)
    )

    def close_splash():
        try:
            splash.destroy()
        except Exception:
            pass

    continue_btn = ttk.Button(
        btn_frame,
        text="Continuar",
        command=close_splash,
        width=18
    )

    continue_btn.pack(
        anchor="center"
    )

    if duration_ms is not None:
        if duration_ms > 0:
            splash.after(
                duration_ms,
                close_splash
            )

    splash.mainloop()


class TrackerLauncher:

    def __init__(self):

        setup_windows_support()

        self.root = tk.Tk()

        set_tk_root(
            self.root
        )

        self.root.title(
            "TrackerLab"
        )

        set_app_icon(
            self.root
        )

        self.root.geometry(
            "860x680"
        )

        self.root.minsize(
            720,
            560
        )

        self.root.resizable(
            True,
            True
        )

        self.video_path = tk.StringVar()

        self.exp_name = tk.StringVar()

        self.tracking_mode = tk.StringVar(
            value="1"
        )

        self.polarity = tk.StringVar(
            value="white_on_black"
        )

        self.apply_bw = tk.BooleanVar(
            value=False
        )

        self.use_clahe = tk.BooleanVar(
            value=False
        )

        self.mutually_exclusive = tk.BooleanVar(
            value=True
        )

        self.csv_base_name = tk.StringVar(
            value=CSV_BASE_NAME
        )

        self.custom_tracking_fps = tk.BooleanVar(
            value=False
        )

        self.tracking_fps = tk.StringVar(
            value=""
        )

        self.result = None

        self._build()

    def _build(self):

        self.root.columnconfigure(
            0,
            weight=1
        )

        self.root.rowconfigure(
            0,
            weight=1
        )

        self.root.rowconfigure(
            1,
            weight=0
        )

        outer = ttk.Frame(
            self.root
        )

        outer.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        outer.columnconfigure(
            0,
            weight=1
        )

        outer.rowconfigure(
            0,
            weight=1
        )

        canvas = tk.Canvas(
            outer,
            highlightthickness=0
        )

        scrollbar = ttk.Scrollbar(
            outer,
            orient="vertical",
            command=canvas.yview
        )

        canvas.configure(
            yscrollcommand=scrollbar.set
        )

        canvas.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        scrollbar.grid(
            row=0,
            column=1,
            sticky="ns"
        )

        frm = ttk.Frame(
            canvas,
            padding=18
        )

        canvas_window = canvas.create_window(
            (0, 0),
            window=frm,
            anchor="nw"
        )

        def on_frame_configure(event=None):
            canvas.configure(
                scrollregion=canvas.bbox(
                    "all"
                )
            )

        def on_canvas_configure(event):
            canvas.itemconfigure(
                canvas_window,
                width=event.width
            )

        def on_mousewheel(event):
            canvas.yview_scroll(
                int(
                    -1 * (
                        event.delta / 120
                    )
                ),
                "units"
            )

        frm.bind(
            "<Configure>",
            on_frame_configure
        )

        canvas.bind(
            "<Configure>",
            on_canvas_configure
        )

        canvas.bind_all(
            "<MouseWheel>",
            on_mousewheel
        )

        ttk.Label(
            frm,
            text="TrackerLab",
            font=(
                "Segoe UI",
                16,
                "bold"
            )
        ).pack(
            anchor="w",
            pady=(0, 14)
        )

        video_row = ttk.Frame(frm)

        video_row.pack(
            fill="x",
            pady=5
        )

        ttk.Label(
            video_row,
            text="Video:",
            width=19
        ).pack(
            side="left"
        )

        ttk.Entry(
            video_row,
            textvariable=self.video_path
        ).pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0, 8)
        )

        ttk.Button(
            video_row,
            text="Seleccionar...",
            command=self.select_video
        ).pack(
            side="left"
        )

        exp_row = ttk.Frame(frm)

        exp_row.pack(
            fill="x",
            pady=7
        )

        ttk.Label(
            exp_row,
            text="Experimento:",
            width=19
        ).pack(
            side="left"
        )

        ttk.Entry(
            exp_row,
            textvariable=self.exp_name
        ).pack(
            side="left",
            fill="x",
            expand=True
        )

        csv_row = ttk.Frame(frm)

        csv_row.pack(
            fill="x",
            pady=7
        )

        ttk.Label(
            csv_row,
            text="Nombre archivos CSV:",
            width=19
        ).pack(
            side="left"
        )

        ttk.Entry(
            csv_row,
            textvariable=self.csv_base_name
        ).pack(
            side="left",
            fill="x",
            expand=True
        )

        fps_row = ttk.Frame(frm)

        fps_row.pack(
            fill="x",
            pady=7
        )

        ttk.Label(
            fps_row,
            text="FPS de tracking:",
            width=19
        ).pack(
            side="left"
        )

        ttk.Checkbutton(
            fps_row,
            text="Modificar",
            variable=self.custom_tracking_fps
        ).pack(
            side="left",
            padx=(0, 8)
        )

        self.fps_spin = ttk.Spinbox(
            fps_row,
            from_=1,
            to=120,
            textvariable=self.tracking_fps,
            width=8
        )

        self.fps_spin.pack(
            side="left"
        )

        ttk.Label(
            fps_row,
            text="(desmarcado = FPS original)"
        ).pack(
            side="left",
            padx=(8, 0)
        )

        mode_row = ttk.Frame(frm)

        mode_row.pack(
            fill="x",
            pady=10
        )

        ttk.Label(
            mode_row,
            text="Modo de tracking:",
            width=19
        ).pack(
            side="left"
        )

        ttk.Radiobutton(
            mode_row,
            text="Diferencia con fondo",
            variable=self.tracking_mode,
            value="1"
        ).pack(
            side="left",
            padx=(0, 18)
        )

        ttk.Radiobutton(
            mode_row,
            text="Silueta",
            variable=self.tracking_mode,
            value="2"
        ).pack(
            side="left",
            padx=(0, 18)
        )

        ttk.Radiobutton(
            mode_row,
            text="Marcador por color",
            variable=self.tracking_mode,
            value="3"
        ).pack(
            side="left"
        )

        pol_row = ttk.Frame(frm)

        pol_row.pack(
            fill="x",
            pady=8
        )

        ttk.Label(
            pol_row,
            text="Polaridad:",
            width=19
        ).pack(
            side="left"
        )

        ttk.Radiobutton(
            pol_row,
            text="Blanco / fondo negro",
            variable=self.polarity,
            value="white_on_black"
        ).pack(
            side="left",
            padx=(0, 18)
        )

        ttk.Radiobutton(
            pol_row,
            text="Negro / fondo blanco",
            variable=self.polarity,
            value="black_on_white"
        ).pack(
            side="left"
        )

        opt_box = ttk.LabelFrame(
            frm,
            text="Opciones",
            padding=10
        )

        opt_box.pack(
            fill="x",
            pady=(12, 10)
        )

        ttk.Checkbutton(
            opt_box,
            text="Aplicar filtro B/N extra",
            variable=self.apply_bw
        ).pack(
            anchor="w"
        )

        ttk.Checkbutton(
            opt_box,
            text="Homogeneizar luz (CLAHE)",
            variable=self.use_clahe
        ).pack(
            anchor="w"
        )

        ttk.Checkbutton(
            opt_box,
            text="Zonas mutuamente excluyentes",
            variable=self.mutually_exclusive
        ).pack(
            anchor="w"
        )

        btns = ttk.Frame(
            self.root,
            padding=(
                18,
                10,
                18,
                14
            )
        )

        btns.grid(
            row=1,
            column=0,
            sticky="ew"
        )

        btns.columnconfigure(
            0,
            weight=1
        )

        ttk.Button(
            btns,
            text="Salir",
            command=self.cancel,
            width=12
        ).grid(
            row=0,
            column=1,
            padx=(0, 8)
        )

        ttk.Button(
            btns,
            text="Iniciar",
            command=self.start,
            width=18
        ).grid(
            row=0,
            column=2
        )

    def select_video(self):

        path = filedialog.askopenfilename(
            parent=self.root,
            title="Seleccionar video",
            filetypes=[
                (
                    "Videos",
                    "*.mp4 *.avi *.mov *.mkv *.wmv *.mpeg *.mpg"
                ),
                (
                    "Todos",
                    "*.*"
                )
            ]
        )

        if not path:
            return

        self.video_path.set(
            path
        )

        base = os.path.splitext(
            os.path.basename(path)
        )[0]

        if not self.exp_name.get().strip():
            self.exp_name.set(
                base
            )

        try:

            cap = cv2.VideoCapture(
                path
            )

            fps = cap.get(
                cv2.CAP_PROP_FPS
            ) or 30

            cap.release()

            fps_max = max(
                1,
                int(round(fps))
            )

            self.fps_spin.configure(
                to=fps_max
            )

            self.tracking_fps.set(
                str(fps_max)
            )

        except Exception:
            pass

    def start(self):

        video = self.video_path.get().strip()

        if not video:
            messagebox.showerror(
                "Falta video",
                "Selecciona un video.",
                parent=self.root
            )
            return

        if not os.path.exists(video):
            messagebox.showerror(
                "Video inválido",
                "El archivo no existe.",
                parent=self.root
            )
            return

        if not self.exp_name.get().strip():
            messagebox.showerror(
                "Falta nombre",
                "Escribe un nombre de experimento.",
                parent=self.root
            )
            return

        tracking_fps = None

        if self.custom_tracking_fps.get():

            try:

                tracking_fps = float(
                    self.tracking_fps.get()
                )

                if tracking_fps <= 0:
                    raise ValueError

            except ValueError:

                messagebox.showerror(
                    "FPS inválido",
                    "Escribe un FPS mayor que 0.",
                    parent=self.root
                )

                return

        self.result = {

            "video_path":
                video,

            "exp_name":
                self.exp_name.get().strip(),

            "csv_base_name":
                self.csv_base_name.get().strip()
                or CSV_BASE_NAME,

            "custom_tracking_fps":
                bool(
                    self.custom_tracking_fps.get()
                ),

            "tracking_fps":
                tracking_fps,

            "tracking_mode":
                self.tracking_mode.get(),

            "polarity":
                self.polarity.get(),

            "apply_bw":
                bool(
                    self.apply_bw.get()
                ),

            "use_clahe":
                bool(
                    self.use_clahe.get()
                ),

            "mutually_exclusive_zones":
                bool(
                    self.mutually_exclusive.get()
                ),

            "root":
                self.root
        }

        self.root.withdraw()

        self.root.quit()

    def cancel(self):

        self.result = None

        self.root.quit()

    def run(self):

        self.root.mainloop()

        return self.result