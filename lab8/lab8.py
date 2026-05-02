import os
import speech_recognition as sr
from gtts import gTTS
import pygame

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain_classic.chains import RetrievalQA
from langchain_classic import PromptTemplate

MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q5_0.gguf"

# Ініціалізація бази знань на основі лекцій та підключення локальної LLM.
def setup_rag_system(filepath="data/lectures.txt"):
    print("[Система]: Завантаження лекційних матеріалів")

    # Завантаження та розбиття тексту
    loader = TextLoader(filepath, encoding='utf-8')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Локальна векторизація тексту
    print("[Система]: Створення векторної бази")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Підключення локальної LLM
    print("[Система]: Завантаження LLM в оперативну пам'ять")
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.1,
        max_tokens=100,
        n_ctx=2048,
        n_threads=10,
        verbose=False
    )

    template = """[INST] Ти помічник для студентів. Використовуй наданий контекст, щоб дати відповідь на питання.

Вимоги до відповіді:
1. Відповідай ВИКЛЮЧНО українською мовою.
2. Відповідь має бути дуже короткою і по суті (максимум 1-2 речення).
3. Якщо в контексті немає інформації, просто скажи "Я не знаю, цієї інформації немає в лекціях".

Контекст лекцій: {context}

Питання студента: {question} [/INST]"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Створення ланцюга RAG
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain

# Перетворення голосу в текст (STT) з використанням мікрофона.
def listen_to_audio():
    recognizer = sr.Recognizer()

    recognizer.pause_threshold = 2.0

    with sr.Microphone() as source:
        print("\n[Система]: Слухаю ваше питання (скажіть 'вихід' для завершення)...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio, language="uk-UA")
        print(f"[Розпізнано]: {text}")
        return text
    except sr.UnknownValueError:
        print("[Помилка]: Не вдалося розпізнати голос. Спробуйте ще раз.")
        return ""
    except sr.RequestError as e:
        print(f"[Помилка]: Помилка сервісу розпізнавання: {e}")
        return ""

# Перетворення тексту в голос (TTS) та його відтворення через pygame.
def speak_text(text):
    print(f"\n[Помічник LLM]: {text}")
    print("[Система]: Озвучую відповідь...")

    tts = gTTS(text=text, lang="uk")
    audio_file = "response.mp3"
    tts.save(audio_file)

    # Відтворення через pygame
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()

    # Чекаємо, поки аудіо дограє до кінця
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    pygame.mixer.quit()

    # Видаляємо тимчасовий файл
    if os.path.exists(audio_file):
        try:
            os.remove(audio_file)
        except PermissionError:
            pass


def main():
    if not os.path.exists("data/lectures.txt"):
        os.makedirs("data", exist_ok=True)
        with open("data/lectures.txt", "w", encoding="utf-8") as f:
            f.write("Лематизація — це процес зведення слова до його базової форми, леми. Вона враховує морфологію. "
                    "Векторизація тексту — це перетворення слів у числові вектори для машинного навчання.")

    if not os.path.exists(MODEL_PATH):
        print(f"[Помилка]: Модель не знайдено за шляхом {MODEL_PATH}.")
        return

    qa_chain = setup_rag_system("data/lectures.txt")
    print("[Система]: Готово! Задавайте питання по курсу NLP.")

    while True:
        query = listen_to_audio()
        if not query:
            continue

        if query.lower() in ["вихід", "стоп", "зупинити", "завершити"]:
            print("[Система]: Завершення роботи.")
            speak_text("Роботу завершено. До побачення!")
            break

        print("[Система]: Генерую відповідь (це може зайняти кілька секунд)...")
        response = qa_chain.invoke(query)

        answer_text = response['result'] if isinstance(response, dict) else response

        speak_text(answer_text)


if __name__ == "__main__":
    main()