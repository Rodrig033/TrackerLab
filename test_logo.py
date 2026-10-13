import tkinter as tk
from PIL import Image, ImageTk
from pathlib import Path

ruta = Path("resources/Tracker_Lab_logo.png").resolve()

print("Ruta utilizada:", ruta)
print("Archivo existente:", ruta.is_file())

# 1. Crear primero la ventana principal.
ventana = tk.Tk()
ventana.title("Prueba del logotipo de TrackerLab")
ventana.geometry("500x550")

# 2. Cargar la imagen.
imagen_original = Image.open(ruta).convert("RGBA")

# 3. Crear la imagen Tkinter asociada a la ventana.
imagen_tk = ImageTk.PhotoImage(
    imagen_original,
    master=ventana
)

print("Dimensiones:", imagen_tk.width(), "x", imagen_tk.height())

# 4. Mostrar el logotipo.
etiqueta = tk.Label(
    ventana,
    image=imagen_tk,
    bg="white"
)
etiqueta.pack(expand=True)

# 5. Mantener viva la referencia a la imagen.
etiqueta.image = imagen_tk

ventana.mainloop()