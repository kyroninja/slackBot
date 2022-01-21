from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.comparisons import LevenshteinDistance, JaccardSimilarity, SpacySimilarity
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.conversation import Statement
from chatterbot.response_selection import get_first_response

import pyttsx3
import speech_recognition as sr

def chatBot():
    chatbot = ChatBot(
    'Buddy',
    storage_adapter = 'chatterbot.storage.SQLStorageAdapter',
    database_uri = 'sqlite:///database_bot.sqlite3',
    logic_adapters=[
        {
            "import_path": "chatterbot.logic.BestMatch",
            "statement_comparison_function": SpacySimilarity,
            "response_selection_method": get_first_response
        },
        {
            "import_path": "chatterbot.logic.MathematicalEvaluation",
            "statement_comparison_function": JaccardSimilarity,
            "response_selection_method": get_first_response
        }
    ]
)
    return chatbot


def trainBot(bot):
    #trainer = ListTrainer(bot) #init trainer
    trainer = ChatterBotCorpusTrainer(bot)
    trainer.train("chatterbot.corpus.english")
    #trainer.train(data) #feed data to bot


def initTTY():
    engine = pyttsx3.init() # object creation
    voices = engine.getProperty('voices')   
    engine.setProperty('rate', 135)     # setting up new voice rate
    engine.setProperty('volume',1.0)    # setting up volume level  between 0 and 1
    engine.setProperty('voice', voices[1].id)   #changing index, changes voices. 1 for female
    return engine

def speak(engine, text):
    engine.say(text) # say text
    engine.runAndWait() # run and clear event queue
    engine.stop() #stop and clear event queue

def initMic():
    r = sr.Recognizer()
    return r

def readMic(mic):
    with sr.Microphone() as source:
        print("Say something: ")
        audio = mic.listen(source)
        return audio

def speechToText(provider, audio, mic):
    if provider == 'google':
        # recognize speech using Google Speech Recognition
        try:
            a = mic.recognize_google(audio)
        except sr.UnknownValueError:
            return 1
        except sr.RequestError as e:
            return 2
        else:
            return a

    if provider == 'sphinx':
        # recognize speech using Google Speech Recognition
        try:
            a = mic.recognize_sphinx(audio)
        except sr.UnknownValueError:
            return 1
        except sr.RequestError as e:
            return 2
        else:
            return a
