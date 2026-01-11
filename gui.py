import tkinter as tk
import pyttsx3
import threading

engine = pyttsx3.init('sapi5')

def speak(text):
    def run():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()

def start_detection():
    print("Starting object detection...")
    speak("Starting object detection mode")
    threading.Thread(target=run_object_detection, daemon=True).start()

def color_mode():
    print("Color mode activated.")
    speak("Color detection mode activated")
    threading.Thread(target=run_color_detection, daemon=True).start()

def distance_mode():
    print("Distance mode activated.")
    speak("Distance mode activated")
    threading.Thread(target=run_distance_detection, daemon=True).start()

# ---- Placeholder functions ----
def run_object_detection():
    print("YOLO detection running...")

def run_color_detection():
    print("Color detection running...")

def run_distance_detection():
    print("Distance detection running...")

# ---- GUI Setup ----
window = tk.Tk()
window.title("VisionAssist")
window.geometry("300x260")
window.resizable(False, False)

title_label = tk.Label(window, text="VisionAssist", font=("Arial", 18, "bold"))
title_label.pack(pady=15)

btn_start = tk.Button(window, text="Start Detection", width=20, height=2, command=start_detection)
btn_start.pack(pady=8)

btn_color = tk.Button(window, text="Color Mode", width=20, height=2, command=color_mode)
btn_color.pack(pady=8)

btn_distance = tk.Button(window, text="Distance Mode", width=20, height=2, command=distance_mode)
btn_distance.pack(pady=8)

window.mainloop()
