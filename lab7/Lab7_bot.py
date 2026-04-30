import pandas as pd
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr
from gtts import gTTS
import os
import pygame
import time

CSV_FILE = 'up_news.csv'
LANG = 'uk'

try:
    nlp = spacy.load("uk_core_news_sm")
except OSError:
    print("Помилка: Модель uk_core_news_sm не знайдено.")
    exit()

# Ініціалізація аудіо-системи
pygame.mixer.init()

# Токенізація, лематизація, видалення стоп-слів та пунктуації
def preprocess_text(text):
    doc = nlp(str(text).lower())
    clean_tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
            clean_tokens.append(token.lemma_)
    return " ".join(clean_tokens)


print("Зчитування даних та обробка NLP конвеєром.")
df = pd.read_csv(CSV_FILE)

# Застосовуємо NLP конвеєр до текстів новин
df['Clean_Text'] = df['Text'].apply(preprocess_text)

# Векторизація тексту (TF-IDF)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Clean_Text'])

# Text-to-Speech
def speak(text):
    print(f"Бот: {text}")
    tts = gTTS(text=text, lang=LANG)
    filename = "response.mp3"
    tts.save(filename)

    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    pygame.mixer.music.unload()
    os.remove(filename)


# Speech-to-Text
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\nСлухаю вас (скажіть 'стоп' для виходу)...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio, language="uk-UA")
            print(f"Ви сказали: {text}")
            return text.lower()
        except sr.WaitTimeoutError:
            print("Час очікування вийшов.")
            return ""
        except sr.UnknownValueError:
            print("Не вдалося розпізнати мову.")
            return ""
        except sr.RequestError as e:
            print(f"Помилка сервісу розпізнавання: {e}")
            return ""

# Знаходить найближчу новину за векторною схожістю
def get_answer(question):
    # NLP обробка запитання
    clean_q = preprocess_text(question)

    # Векторизація запитання
    q_vec = vectorizer.transform([clean_q])

    # Обчислення косинусної подібності між питанням та всіма новинами
    similarities = cosine_similarity(q_vec, tfidf_matrix).flatten()

    best_match_idx = np.argmax(similarities)
    best_score = similarities[best_match_idx]

    # Якщо схожість дуже низька, бот нічого не знайшов в інфопросторі
    if best_score < 0.05:
        return "На жаль, в моїй базі останніх новин немає інформації за вашим запитом."

    # Беремо перше речення з оригінального тексту
    found_news = str(df.iloc[best_match_idx]['Text'])

    # Генерація короткої відповіді
    sentences = found_news.split('.')
    summary = ".".join(sentences[:2]) + "."

    return f"За вашим запитом знайшла таку новину: {summary}"

def run_bot():
    speak("Привіт! Я ваш аудіо-помічник. Запитайте мене про останні новини.")

    while True:
        user_question = listen()

        if not user_question:
            continue

        if "стоп" in user_question or "вихід" in user_question or "до побачення" in user_question:
            speak("До зустрічі! Була рада допомогти.")
            break

        answer = get_answer(user_question)
        speak(answer)


if __name__ == "__main__":
    run_bot()