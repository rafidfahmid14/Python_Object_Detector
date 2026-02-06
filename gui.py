import tkinter as tk
import pyttsx3
import threading
import queue

# Try to import the detector run functions; provide safe fallbacks if unavailable
try:
    from object_detection import run_object_detection
except Exception:
    def run_object_detection(stop_event=None):
        print("object_detection module not available")

try:
    from color_detection import run_color_detection
except Exception:
    def run_color_detection(stop_event=None):
        print("color_detection module not available")

_speech_queue = queue.Queue()

def _speech_worker():
    engine = pyttsx3.init('sapi5')
    while True:
        text = _speech_queue.get()
        if text is None:
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception:
            # ignore speech errors and continue
            pass
    try:
        engine.stop()
    except Exception:
        pass

_speech_thread = threading.Thread(target=_speech_worker, daemon=True)
_speech_thread.start()

def speak(text):
    _speech_queue.put(text)

def start_detection():
    print("Starting object detection...")
    speak("Starting object detection mode")
    threading.Thread(target=lambda: run_object_detection(), daemon=True).start()

def color_mode():
    print("Color mode activated.")
    speak("Color detection mode activated")
    threading.Thread(target=lambda: run_color_detection(), daemon=True).start()

# ---- Note: run functions are imported above ----

# ---- GUI Setup ----
window = tk.Tk()
window.title("VisionAssist")
window.geometry("300x200")
window.resizable(False, False)

title_label = tk.Label(window, text="VisionAssist", font=("Arial", 18, "bold"))
title_label.pack(pady=15)

btn_start = tk.Button(window, text="Start Detection", width=20, height=2, command=start_detection)
btn_start.pack(pady=8)

btn_color = tk.Button(window, text="Color Mode", width=20, height=2, command=color_mode)
btn_color.pack(pady=8)

def on_closing():
    # signal the speech worker to exit and wait briefly
    try:
        _speech_queue.put(None)
    except Exception:
        pass
    try:
        _speech_thread.join(timeout=1.0)
    except Exception:
        pass
    window.destroy()

window.protocol("WM_DELETE_WINDOW", on_closing)

window.mainloop()
