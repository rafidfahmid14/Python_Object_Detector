import time
import pyttsx3
import threading

engine_lock = threading.Lock()

# speak function is a wrapper created based on issue with pyttsx not working with multiple calls to say().
# added thread with lock to ensure the calls do not overlap to resolve issue with 'run loop already started' error
def speak(text):
    def run():
        with engine_lock:
            engine = pyttsx3.init('sapi5')
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine
    
    threading.Thread(target=run, daemon=True).start()