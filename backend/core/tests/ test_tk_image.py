import tkinter as tk

root = tk.Tk()
root.title("Diagnóstico gráfico de TrackerLab")
root.geometry("500x400")

canvas = tk.Canvas(
    root,
    width=400,
    height=300,
    bg="white",
    highlightthickness=0
)

canvas.pack(expand=True)

# Figura geométrica generada directamente por Tkinter.
canvas.create_rectangle(
    50, 50, 200, 200,
    fill="#ff00ff",
    outline="black",
    width=3
)

canvas.create_text(
    125, 125,
    text="TRACKERLAB",
    fill="black",
    font=("Arial", 16, "bold")
)

# Imagen completamente nueva generada en memoria.
image = tk.PhotoImage(width=100, height=100)
image.put("#00ff00", to=(0, 0, 100, 100))

label = tk.Label(
    root,
    image=image,
    bg="white"
)

label.image = image
label.pack(pady=10)

print("Tkinter:", root.tk.call("info", "patchlevel"))
print("Ventana de diagnóstico abierta.")

root.mainloop()