import pyttsx3

# Initialize TTS engine (Windows SAPI5)
engine = pyttsx3.init('sapi5')

# Optional: list available voices
voices = engine.getProperty('voices')
for i, voice in enumerate(voices):
    print(i, voice.name)

# Optional: choose voice (0 or 1 usually)
engine.setProperty('voice', voices[0].id)

# Optional: adjust speed and volume
engine.setProperty('rate', 170)     # default ~200
engine.setProperty('volume', 1.0)   # 0.0 to 1.0

# Speak test sentence
engine.say("Red chair in front of you")
engine.runAndWait()
