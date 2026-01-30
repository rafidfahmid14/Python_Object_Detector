import tkinter as tk
import pyttsx3

# --- Initialize Text-to-Speech ---
engine = pyttsx3.init('sapi5')

def speak(text):
    engine.say(text)
    engine.runAndWait()


# --- Button Functions ---
def start_detection():
    print("Starting object detection...")
    speak("Starting object detection mode")
    # later: call your object detection code here


def color_mode():
    print("Color mode activated.")
    speak("Color detection mode activated")
    # later: run your color detection code


def distance_mode():
    print("Distance mode activated.")
    speak("Distance mode activated")
    # later: your distance algorithm


# --- GUI Setup ---
window = tk.Tk()
window.title("VisionAssist")
window.geometry("300x250")

title_label = tk.Label(window, text="VisionAssist", font=("Arial", 18, "bold"))
title_label.pack(pady=15)

btn_start = tk.Button(window, text="Start Detection", command=start_detection, width=20, height=2)
btn_start.pack(pady=10)

btn_color = tk.Button(window, text="Color Mode", command=color_mode, width=20, height=2)
btn_color.pack(pady=10)

btn_distance = tk.Button(window, text="Distance Mode", command=distance_mode, width=20, height=2)
btn_distance.pack(pady=10)

window.mainloop()
