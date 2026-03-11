import tkinter as tk
from PIL import Image, ImageTk
import cairosvg
import io
import main
import add_data
import visualise
from tkinter import messagebox

def button_action(section):
    # take the value and call the function
    if section == "Photo":
        value = entry_vars[section].get()
        main.photo(value)
    elif section == "Video":
        value = entry_vars[section].get()
        messagebox.showinfo('Work in progress', 'Press ok to process the video. It might take some time.')
        main.video(value)
        add_data.use_in_final()
        visualise.visual(value)
    else:
        main.camera()


# Create the main application window
root = tk.Tk()
root.title("ANPR - Automatic Number Plate Recognition")
root.geometry("1240x620")
root.configure(bg="#ADD8E6")

entry_vars = {}
sections = ["Photo", "Video", "Real Time"]
image_paths = ["icons/photo.svg", "icons/video.svg", "icons/camera.svg"]

images = []

heading_label = tk.Label(root, text="License Plate Recognition System", font=("Times New Roman", 28, "bold"), bg="#ADD8E6", fg="#333")
heading_label.pack(pady=(120, 20))

main_frame = tk.Frame(root, bg="#ADD8E6")
main_frame.pack(pady=(0, 180))


def create_rounded_rectangle(canvas, x1, y1, x2, y2, radius=20, **kwargs):
    points = [
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1, x2, y1 + radius,
        x2, y2 - radius,
        x2, y2, x2 - radius, y2,
        x1 + radius, y2,
        x1, y2, x1, y2 - radius,
        x1, y1 + radius,
        x1, y1
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


for i, section in enumerate(sections):
    canvas = tk.Canvas(main_frame, width=300, height=240, bg="#ADD8E6", highlightthickness=0)
    canvas.pack(side=tk.LEFT, padx=20, pady=10)

    create_rounded_rectangle(canvas, 5, 5, 295, 235, radius=30, fill="#B0E0E6", outline="#ADD8E6")

    frame = tk.Frame(canvas, bg="#B0E0E6")
    frame.place(x=20, y=20, width=260, height=200)

    # Convert and display SVG image
    with open(image_paths[i], "rb") as svg_file:
        png_data = cairosvg.svg2png(bytestring=svg_file.read())
    image = Image.open(io.BytesIO(png_data))
    image = image.resize((50, 50), Image.Resampling.LANCZOS)
    img = ImageTk.PhotoImage(image)
    images.append(img)  # Keep a reference
    image_label = tk.Label(frame, image=img, bg="#B0E0E6")
    image_label.pack()

    heading = tk.Label(frame, text=section, font=("Times New Roman", 16, "bold"), bg="#B0E0E6", fg="#333")

    if section == "Real Time":
        heading.pack(pady=(0, 20))
    else:
        heading.pack()

    if section == "Photo" or section == "Video":
        entry_vars[section] = tk.StringVar()
        entry_canvas = tk.Canvas(frame, width=260, height=50, bg="#B0E0E6", highlightthickness=0)
        entry_canvas.pack(pady=(15, 5))
        create_rounded_rectangle(entry_canvas, 2, 2, 258, 48, radius=30, fill="#E0FFFF", outline="#00008B")
        entry = tk.Entry(frame, textvariable=entry_vars[section], width=30, font=("Times New Roman", 12), bg="#E0FFFF", fg="#333", bd=0)
        entry_canvas.create_window(130, 28, window=entry)

    button_canvas = tk.Canvas(frame, width=120, height=40, bg="#B0E0E6", highlightthickness=0)
    button_canvas.pack()
    create_rounded_rectangle(button_canvas, 2, 2, 118, 38, radius=30, fill="#4682B4", outline="#00008B")

    if section == 3:
        button = tk.Button(frame, text="Open Cam", command=lambda sec=section: button_action(sec), bg="#4682B4",
                           fg="white", font=("Times New Roman", 12),
                           bd=0, highlightthickness=0, activebackground="#5A9BD4")
    else:
        button = tk.Button(frame, text="Submit", command=lambda sec=section: button_action(sec), bg="#4682B4",
                           fg="white", font=("Times New Roman", 12),
                           bd=0, highlightthickness=0, activebackground="#5A9BD4")

    button_canvas.create_window(60, 20, window=button)
    button_canvas.bind('<Button-1>', lambda event, sec=section: button_action(sec))


root.mainloop()
