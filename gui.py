import tkinter as tk

window = tk.Tk()
window.title("VisionAssist")

btn_start = tk.Button(window, text="Start Detection")
btn_start.pack(pady=10)

btn_color = tk.Button(window, text="Color Mode")
btn_color.pack(pady=10)

btn_distance = tk.Button(window, text="Distance Mode")
btn_distance.pack(pady=10)

window.mainloop()
