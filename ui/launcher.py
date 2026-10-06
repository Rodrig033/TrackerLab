import os
import sys
import platform
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2

from ui.dialogs import set_tk_root


CSV_BASE_NAME = "track_RATA"
APP_NAME = "TrackerLab"
APP_VERSION = "1.0"


def setup_platform_support():
    if platform.system() == "Windows":
        try:
            import ctypes

            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                pass

            try:
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                    "TrackerLab.2026"
                )
            except Exception:
                pass

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
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.dirname(sys.executable)
    else:
        base_path = project_directory()

    return os.path.join(
        base_path,
        "resources",
        filename
    )


def set_app_icon(window):
    png_path = resource_path(
        "Tracker_Lab_logo.png"
    )

    ico_path = resource_path(
        "Tracker_Lab_icon.ico"
    )

    if platform.system() == "Windows":
        try:
            if os.path.exists(ico_path):
                window.iconbitmap(
                    ico_path
                )
        except Exception:
            pass

    try:
        if os.path.exists(png_path):
            icon_image = tk.PhotoImage(
                file=png_path
            )

            window.iconphoto(
                True,
                icon_image
            )

            window._trackerlab_icon = icon_image

    except Exception:
        pass


def open_folder(path):
    os.makedirs(
        path,
        exist_ok=True
    )

    system = platform.system()

    try:
        if system == "Windows":
            os.startfile(path)

        elif system == "Darwin":
            subprocess.Popen(
                [
                    "open",
                    path
                ]
            )

        else:
            subprocess.Popen(
                [
                    "xdg-open",
                    path
                ]
            )

        return True

    except Exception:
        return False


def show_splash_screen(duration_ms=1600):
    splash = tk.Tk()

    splash.title(
        APP_NAME
    )

    set_app_icon(
        splash
    )

    splash.configure(
        bg="white"
    )

    splash.resizable(
        False,
        False
    )

    width = 500
    height = 400

    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()

    x = max(
        0,
        (screen_width - width) // 2
    )

    y = max(
        0,
        (screen_height - height) // 2
    )

    splash.geometry(
        f"{width}x{height}+{x}+{y}"
    )

    container = tk.Frame(
        splash,
        bg="white",
        padx=30,
        pady=30
    )

    container.pack(
        fill="both",
        expand=True
    )

    logo_path = resource_path(
        "Tracker_Lab_logo.png"
    )

    try:
        logo = tk.PhotoImage(
            file=logo_path
        )

        factor = max(
            1,
            int(
                max(
                    logo.width(),
                    logo.height()
                ) / 150
            )
        )

        if factor > 1:
            logo = logo.subsample(
                factor,
                factor
            )

        logo_label = tk.Label(
            container,
            image=logo,
            bg="white"
        )

        logo_label.image = logo

        logo_label.pack(
            pady=(10, 15)
        )

    except Exception:
        pass

    tk.Label(
        container,
        text=APP_NAME,
        bg="white",
        fg="#202020",
        font=(
            "Arial",
            24,
            "bold"
        )
    ).pack()

    tk.Label(
        container,
        text="Seguimiento y análisis experimental",
        bg="white",
        fg="#555555",
        font=(
            "Arial",
            11
        )
    ).pack(
        pady=(6, 18)
    )

    tk.Label(
        container,
        text=f"Versión {APP_VERSION}",
        bg="white",
        fg="#777777",
        font=(
            "Arial",
            9
        )
    ).pack()

    tk.Label(
        container,
        text="Universidad Veracruzana · 2026",
        bg="white",
        fg="#777777",
        font=(
            "Arial",
            9
        )
    ).pack(
        pady=(4, 0)
    )

    tk.Label(
        container,
        text="Iniciando...",
        bg="white",
        fg="#888888",
        font=(
            "Arial",
            9
        )
    ).pack(
        side="bottom",
        pady=(20, 0)
    )

    splash.after(
        duration_ms,
        splash.destroy
    )

    splash.mainloop()


class TrackerLauncher:

    def __init__(self):
        setup_platform_support()

        self.root = tk.Tk()

        set_tk_root(
            self.root
        )

        self.root.title(
            APP_NAME
        )

        set_app_icon(
            self.root
        )

        self.root.geometry(
            "920x760"
        )

        self.root.minsize(
            780,
            620
        )

        self.root.resizable(
            True,
            True
        )

        self.video_path = tk.StringVar()

        self.exp_name = tk.StringVar()

        self.csv_base_name = tk.StringVar(
            value=CSV_BASE_NAME
        )

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

        self.custom_tracking_fps = tk.BooleanVar(
            value=False
        )

        self.tracking_fps = tk.StringVar(
            value=""
        )

        self.video_info = tk.StringVar(
            value="No se ha seleccionado ningún video."
        )

        self.status_text = tk.StringVar(
            value="Listo"
        )

        self.result = None

        self._create_styles()
        self._create_menu()
        self._build()
        self._update_fps_state()


    def _create_styles(self):
        style = ttk.Style()

        themes = style.theme_names()

        system = platform.system()

        if system == "Windows":
            preferred = [
                "vista",
                "xpnative",
                "clam"
            ]

        elif system == "Darwin":
            preferred = [
                "aqua",
                "clam"
            ]

        else:
            preferred = [
                "clam",
                "alt",
                "default"
            ]

        for theme in preferred:
            if theme in themes:
                try:
                    style.theme_use(
                        theme
                    )
                    break
                except Exception:
                    pass

        style.configure(
            "Title.TLabel",
            font=(
                "Arial",
                22,
                "bold"
            )
        )

        style.configure(
            "Subtitle.TLabel",
            font=(
                "Arial",
                10
            )
        )

        style.configure(
            "Section.TLabelframe.Label",
            font=(
                "Arial",
                10,
                "bold"
            )
        )

        style.configure(
            "Primary.TButton",
            font=(
                "Arial",
                10,
                "bold"
            ),
            padding=8
        )


    def _create_menu(self):
        menu_bar = tk.Menu(
            self.root
        )

        file_menu = tk.Menu(
            menu_bar,
            tearoff=0
        )

        file_menu.add_command(
            label="Abrir video",
            command=self.select_video
        )

        file_menu.add_separator()

        file_menu.add_command(
            label="Salir",
            command=self.cancel
        )

        menu_bar.add_cascade(
            label="Archivo",
            menu=file_menu
        )

        experiment_menu = tk.Menu(
            menu_bar,
            tearoff=0
        )

        experiment_menu.add_command(
            label="Nuevo experimento",
            command=self.new_experiment
        )

        experiment_menu.add_command(
            label="Abrir carpeta de experimentos",
            command=self.open_experiments_folder
        )

        menu_bar.add_cascade(
            label="Experimento",
            menu=experiment_menu
        )

        help_menu = tk.Menu(
            menu_bar,
            tearoff=0
        )

        help_menu.add_command(
            label="Acerca de TrackerLab",
            command=self.show_about
        )

        menu_bar.add_cascade(
            label="Ayuda",
            menu=help_menu
        )

        self.root.config(
            menu=menu_bar
        )


    def _build(self):
        self.root.columnconfigure(
            0,
            weight=1
        )

        self.root.rowconfigure(
            0,
            weight=1
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

        container = ttk.Frame(
            canvas,
            padding=22
        )

        canvas_window = canvas.create_window(
            (0, 0),
            window=container,
            anchor="nw"
        )

        def update_scroll_region(event=None):
            canvas.configure(
                scrollregion=canvas.bbox(
                    "all"
                )
            )

        def resize_container(event):
            canvas.itemconfigure(
                canvas_window,
                width=event.width
            )

        def mousewheel(event):
            if platform.system() != "Linux":
                canvas.yview_scroll(
                    int(
                        -1
                        * (
                            event.delta / 120
                        )
                    ),
                    "units"
                )

        def linux_scroll_up(event):
            canvas.yview_scroll(
                -1,
                "units"
            )

        def linux_scroll_down(event):
            canvas.yview_scroll(
                1,
                "units"
            )

        container.bind(
            "<Configure>",
            update_scroll_region
        )

        canvas.bind(
            "<Configure>",
            resize_container
        )

        canvas.bind_all(
            "<MouseWheel>",
            mousewheel
        )

        canvas.bind_all(
            "<Button-4>",
            linux_scroll_up
        )

        canvas.bind_all(
            "<Button-5>",
            linux_scroll_down
        )

        self._build_header(
            container
        )

        self._build_video_section(
            container
        )

        self._build_tracking_section(
            container
        )

        self._build_options_section(
            container
        )

        self._build_buttons(
            container
        )

        self._build_status_bar()


    def _build_header(self, parent):
        header = ttk.Frame(
            parent
        )

        header.pack(
            fill="x",
            pady=(0, 20)
        )

        logo_path = resource_path(
            "Tracker_Lab_logo.png"
        )

        try:
            logo = tk.PhotoImage(
                file=logo_path
            )

            factor = max(
                1,
                int(
                    max(
                        logo.width(),
                        logo.height()
                    ) / 75
                )
            )

            if factor > 1:
                logo = logo.subsample(
                    factor,
                    factor
                )

            logo_label = ttk.Label(
                header,
                image=logo
            )

            logo_label.image = logo

            logo_label.pack(
                side="left",
                padx=(0, 15)
            )

        except Exception:
            pass

        text_frame = ttk.Frame(
            header
        )

        text_frame.pack(
            side="left",
            fill="x",
            expand=True
        )

        ttk.Label(
            text_frame,
            text=APP_NAME,
            style="Title.TLabel"
        ).pack(
            anchor="w"
        )

        ttk.Label(
            text_frame,
            text="Sistema de seguimiento y análisis experimental",
            style="Subtitle.TLabel"
        ).pack(
            anchor="w",
            pady=(3, 0)
        )

        ttk.Label(
            header,
            text=f"v{APP_VERSION}"
        ).pack(
            side="right",
            anchor="n"
        )


    def _build_video_section(self, parent):
        box = ttk.LabelFrame(
            parent,
            text="Video y experimento",
            padding=16,
            style="Section.TLabelframe"
        )

        box.pack(
            fill="x",
            pady=(0, 14)
        )

        box.columnconfigure(
            1,
            weight=1
        )

        ttk.Label(
            box,
            text="Video:"
        ).grid(
            row=0,
            column=0,
            sticky="w",
            pady=6
        )

        ttk.Entry(
            box,
            textvariable=self.video_path
        ).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=10
        )

        ttk.Button(
            box,
            text="Examinar...",
            command=self.select_video
        ).grid(
            row=0,
            column=2
        )

        ttk.Label(
            box,
            textvariable=self.video_info
        ).grid(
            row=1,
            column=1,
            columnspan=2,
            sticky="w",
            padx=10,
            pady=(0, 12)
        )

        ttk.Label(
            box,
            text="Experimento:"
        ).grid(
            row=2,
            column=0,
            sticky="w",
            pady=6
        )

        ttk.Entry(
            box,
            textvariable=self.exp_name
        ).grid(
            row=2,
            column=1,
            columnspan=2,
            sticky="ew",
            padx=10
        )

        ttk.Label(
            box,
            text="Nombre CSV:"
        ).grid(
            row=3,
            column=0,
            sticky="w",
            pady=6
        )

        ttk.Entry(
            box,
            textvariable=self.csv_base_name
        ).grid(
            row=3,
            column=1,
            columnspan=2,
            sticky="ew",
            padx=10
        )


    def _build_tracking_section(self, parent):
        box = ttk.LabelFrame(
            parent,
            text="Método de seguimiento",
            padding=16,
            style="Section.TLabelframe"
        )

        box.pack(
            fill="x",
            pady=(0, 14)
        )

        ttk.Radiobutton(
            box,
            text="Diferencia con fondo",
            variable=self.tracking_mode,
            value="1"
        ).pack(
            anchor="w"
        )

        ttk.Label(
            box,
            text="Recomendado cuando la cámara y el fondo permanecen estables."
        ).pack(
            anchor="w",
            padx=24,
            pady=(0, 8)
        )

        ttk.Radiobutton(
            box,
            text="Silueta",
            variable=self.tracking_mode,
            value="2"
        ).pack(
            anchor="w"
        )

        ttk.Label(
            box,
            text="Detecta al sujeto utilizando su forma o silueta."
        ).pack(
            anchor="w",
            padx=24,
            pady=(0, 8)
        )

        ttk.Radiobutton(
            box,
            text="Marcador por color",
            variable=self.tracking_mode,
            value="3"
        ).pack(
            anchor="w"
        )

        ttk.Label(
            box,
            text="Realiza seguimiento utilizando un rango de color."
        ).pack(
            anchor="w",
            padx=24
        )


    def _build_options_section(self, parent):
        box = ttk.LabelFrame(
            parent,
            text="Opciones",
            padding=16,
            style="Section.TLabelframe"
        )

        box.pack(
            fill="x",
            pady=(0, 14)
        )

        fps_frame = ttk.Frame(
            box
        )

        fps_frame.pack(
            fill="x",
            pady=(0, 10)
        )

        ttk.Checkbutton(
            fps_frame,
            text="Modificar FPS de procesamiento",
            variable=self.custom_tracking_fps,
            command=self._update_fps_state
        ).pack(
            side="left"
        )

        ttk.Label(
            fps_frame,
            text="FPS:"
        ).pack(
            side="left",
            padx=(25, 5)
        )

        self.fps_spin = ttk.Spinbox(
            fps_frame,
            from_=1,
            to=240,
            width=8,
            textvariable=self.tracking_fps
        )

        self.fps_spin.pack(
            side="left"
        )

        ttk.Separator(
            box
        ).pack(
            fill="x",
            pady=10
        )

        ttk.Label(
            box,
            text="Apariencia del sujeto"
        ).pack(
            anchor="w",
            pady=(0, 6)
        )

        ttk.Radiobutton(
            box,
            text="Sujeto claro sobre fondo oscuro",
            variable=self.polarity,
            value="white_on_black"
        ).pack(
            anchor="w"
        )

        ttk.Radiobutton(
            box,
            text="Sujeto oscuro sobre fondo claro",
            variable=self.polarity,
            value="black_on_white"
        ).pack(
            anchor="w"
        )

        ttk.Separator(
            box
        ).pack(
            fill="x",
            pady=10
        )

        ttk.Checkbutton(
            box,
            text="Homogeneizar iluminación (CLAHE)",
            variable=self.use_clahe
        ).pack(
            anchor="w",
            pady=3
        )

        ttk.Checkbutton(
            box,
            text="Aplicar filtro blanco y negro adicional",
            variable=self.apply_bw
        ).pack(
            anchor="w",
            pady=3
        )

        ttk.Checkbutton(
            box,
            text="Usar zonas mutuamente excluyentes",
            variable=self.mutually_exclusive
        ).pack(
            anchor="w",
            pady=3
        )


    def _build_buttons(self, parent):
        frame = ttk.Frame(
            parent
        )

        frame.pack(
            fill="x",
            pady=(8, 5)
        )

        ttk.Button(
            frame,
            text="Cancelar",
            command=self.cancel
        ).pack(
            side="right",
            padx=(10, 0)
        )

        ttk.Button(
            frame,
            text="Iniciar análisis",
            command=self.start,
            style="Primary.TButton"
        ).pack(
            side="right"
        )


    def _build_status_bar(self):
        status = ttk.Frame(
            self.root,
            padding=(
                12,
                6
            )
        )

        status.grid(
            row=1,
            column=0,
            sticky="ew"
        )

        ttk.Separator(
            status
        ).pack(
            fill="x",
            pady=(0, 6)
        )

        ttk.Label(
            status,
            textvariable=self.status_text
        ).pack(
            side="left"
        )

        ttk.Label(
            status,
            text=f"{APP_NAME} {APP_VERSION}"
        ).pack(
            side="right"
        )


    def _update_fps_state(self):
        if self.custom_tracking_fps.get():
            self.fps_spin.configure(
                state="normal"
            )
        else:
            self.fps_spin.configure(
                state="disabled"
            )


    def select_video(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Seleccionar video",
            filetypes=[
                (
                    "Videos",
                    "*.mp4 *.avi *.mov *.mkv *.wmv *.mpeg *.mpg *.m4v"
                ),
                (
                    "Todos los archivos",
                    "*.*"
                )
            ]
        )

        if not path:
            return

        path = os.path.abspath(
            path
        )

        self.video_path.set(
            path
        )

        video_name = os.path.splitext(
            os.path.basename(
                path
            )
        )[0]

        if not self.exp_name.get().strip():
            self.exp_name.set(
                video_name
            )

        self.load_video_information(
            path
        )

        self.status_text.set(
            "Video cargado correctamente"
        )


    def load_video_information(self, path):
        cap = cv2.VideoCapture(
            path
        )

        if not cap.isOpened():
            self.video_info.set(
                "No se pudo obtener información del video."
            )

            return

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        frames = cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        cap.release()

        if not fps or fps <= 0:
            fps = 30.0

        if not frames or frames < 0:
            frames = 0

        duration = (
            frames / fps
            if fps > 0
            else 0
        )

        hours = int(
            duration // 3600
        )

        minutes = int(
            (duration % 3600) // 60
        )

        seconds = int(
            duration % 60
        )

        if hours > 0:
            duration_text = (
                f"{hours:02d}:"
                f"{minutes:02d}:"
                f"{seconds:02d}"
            )

        else:
            duration_text = (
                f"{minutes:02d}:"
                f"{seconds:02d}"
            )

        fps_limit = max(
            1,
            int(
                round(fps)
            )
        )

        self.tracking_fps.set(
            str(fps_limit)
        )

        self.fps_spin.configure(
            to=fps_limit
        )

        self.video_info.set(
            f"{width} × {height}"
            f" · {fps:.2f} FPS"
            f" · {duration_text}"
            f" · {int(frames)} frames"
        )


    def new_experiment(self):
        self.exp_name.set(
            ""
        )

        self.status_text.set(
            "Nuevo experimento"
        )


    def open_experiments_folder(self):
        try:
            from persistence.experiment_config import (
                EXPERIMENTS_DIR
            )

            folder = EXPERIMENTS_DIR

        except Exception:
            folder = os.path.join(
                project_directory(),
                "Experimentos"
            )

        if not open_folder(
            folder
        ):
            messagebox.showinfo(
                "Carpeta de experimentos",
                folder,
                parent=self.root
            )


    def show_about(self):
        messagebox.showinfo(
            "Acerca de TrackerLab",
            f"{APP_NAME}\n\n"
            f"Versión {APP_VERSION}\n\n"
            "Sistema de seguimiento y análisis experimental.\n\n"
            "Compatible con Windows, macOS y Linux.\n\n"
            "Universidad Veracruzana · 2026",
            parent=self.root
        )


    def start(self):
        video = self.video_path.get().strip()

        if not video:
            messagebox.showerror(
                "Falta video",
                "Selecciona un video antes de iniciar.",
                parent=self.root
            )

            return

        if not os.path.exists(
            video
        ):
            messagebox.showerror(
                "Video inválido",
                "El archivo seleccionado no existe.",
                parent=self.root
            )

            return

        experiment = self.exp_name.get().strip()

        if not experiment:
            messagebox.showerror(
                "Falta experimento",
                "Escribe un nombre para el experimento.",
                parent=self.root
            )

            return

        tracking_fps_value = None

        if self.custom_tracking_fps.get():
            try:
                tracking_fps_value = float(
                    self.tracking_fps.get().strip()
                )

                if tracking_fps_value <= 0:
                    raise ValueError

            except Exception:
                messagebox.showerror(
                    "FPS inválido",
                    "Escribe un FPS válido mayor que 0.",
                    parent=self.root
                )

                return

        self.result = {
            "video_path":
                video,

            "exp_name":
                experiment,

            "csv_base_name":
                self.csv_base_name.get().strip()
                or CSV_BASE_NAME,

            "custom_tracking_fps":
                bool(
                    self.custom_tracking_fps.get()
                ),

            "tracking_fps":
                tracking_fps_value,

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

        self.status_text.set(
            "Iniciando análisis..."
        )

        self.root.withdraw()
        self.root.quit()


    def cancel(self):
        self.result = None

        try:
            self.root.quit()
        except Exception:
            pass


    def run(self):
        self.root.mainloop()

        return self.result