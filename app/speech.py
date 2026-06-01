import pyttsx3
import threading

engine = pyttsx3.init()
engine.setProperty('rate', 160)


def speak_text(text: str):

    def run():
        engine.say(text)
        engine.runAndWait()

    thread = threading.Thread(target=run)
    thread.start()