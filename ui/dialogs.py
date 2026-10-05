import tkinter as tk
from tkinter import messagebox, simpledialog


_tk_root = None


def get_tk_root():
    global _tk_root

    if _tk_root is None:
        _tk_root = tk.Tk()
        _tk_root.withdraw()

        try:
            _tk_root.attributes("-topmost", True)
        except Exception:
            pass

    return _tk_root


def set_tk_root(root):
    global _tk_root
    _tk_root = root


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


def gui_ask_yes_no(title, msg, default_yes=True):
    try:
        return bool(
            messagebox.askyesno(
                title,
                msg,
                parent=get_tk_root(),
                default="yes" if default_yes else "no"
            )
        )
    except Exception:
        return default_yes


def gui_ask_string(title, prompt, default=""):
    try:
        ans = simpledialog.askstring(
            title,
            prompt,
            initialvalue=default,
            parent=get_tk_root()
        )

        if ans is None:
            return None

        return str(ans).strip()

    except Exception:
        return default


def gui_ask_float(title, prompt, default=None):
    while True:
        ans = gui_ask_string(
            title,
            prompt,
            "" if default is None else str(default)
        )

        if ans is None or ans == "":
            return default

        try:
            return float(ans)

        except ValueError:
            gui_warning(
                "Valor inválido",
                "Escribe un número válido."
            )


def ask_yes_no(prompt, default_yes=True):
    return gui_ask_yes_no(
        "Confirmar",
        prompt,
        default_yes
    )