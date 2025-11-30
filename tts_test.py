import pyttsx3

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
print(voices)

engine.say("red chair in front of u")
engine.runAndWait()


