import tkinter as tk
import threading
from combined_manager import CombinedManager
from common import speak


# --- Button Functions ---
_manager = None

def start_detection():
    global _manager
    print("Starting object detection...")
    speak("Starting object detection mode")
    speak("Testing 123")
    speak("Starting object detection mode again")
    if _manager is None:
        _manager = CombinedManager()
        try:
            _manager.start()
        except Exception as e:
            print("Failed to start camera:", e)
            speak("Failed to start camera")
            _manager = None
            return
    _manager.enable_object(True)

def color_mode():
    global _manager
    print("Color mode activated.")
    speak("Color detection mode activated")
    if _manager is None:
        _manager = CombinedManager()
        try:
            _manager.start()
        except Exception as e:
            print("Failed to start camera:", e)
            speak("Failed to start camera")
            _manager = None
            return
    _manager.enable_color(True)

# ---- Placeholder (wrappers that call module functions) ----
def run_object_detection():
    # kept for backward compatibility; starts combined manager object mode
    start_detection()

def run_color_detection():
    # kept for backward compatibility; starts combined manager color mode
    color_mode()

# ---- GUI Setup ----
window = tk.Tk()
window.title("VisionAssist")
window.geometry("300x200")
window.resizable(False, False)

title_label = tk.Label(window, text="VisionAssist", font=("Arial", 18, "bold"))
title_label.pack(pady=15)

btn_start = tk.Button(window, text="Start Detection", command=start_detection, width=20, height=2)
btn_start.pack(pady=10)

btn_color = tk.Button(window, text="Color Mode", command=color_mode, width=20, height=2)
btn_color.pack(pady=10)

window.mainloop()
