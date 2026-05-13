import os
import sys
import json
import logging
from datetime import datetime
import pyttsx3
import speech_recognition as sr
from dotenv import load_dotenv

load_dotenv()

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("nlp_assistant.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
# Вимикаємо зайві логи в консолі для чистоти інтерфейсу
logging.getLogger().setLevel(logging.WARNING)


# Модуль конфігурації
class Config:
    TTS_RATE = 150
    TTS_VOLUME = 0.9
    STT_TIMEOUT = 7
    STT_PHRASE_LIMIT = 20
    STT_LANGUAGE = 'uk-UA'
    HISTORY_DIR = "dialog_history"

    SYSTEM_PROMPT = (
        "Ти професійний NLP-Асистент. Твоя мета допомагати у вивченні "
        "технологій обробки природної мови. Відповідай лаконічно, українською мовою. "
        "Для технічних питань надавай приклади коду на Python."
    )


# Модуль обробки аудіо (STT & TTS)
class AudioProcessor:
    def __init__(self):
        self.tts_engine = self._init_tts()
        self.recognizer = sr.Recognizer()

        # Налаштування чутливості мікрофона
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 300

    def _init_tts(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', Config.TTS_RATE)
            engine.setProperty('volume', Config.TTS_VOLUME)
            return engine
        except Exception as e:
            logging.error(f"Помилка ініціалізації TTS: {e}")
            return None

    def speak(self, text: str):
        if not self.tts_engine:
            return
        # Базове очищення тексту від спецсимволів перед озвученням
        clean_text = text.replace('```python', '').replace('```', '').replace('#', '')
        # Обмежуємо довжину тексту для озвучення, щоб уникнути зависань
        short_text = clean_text[:600] + ("..." if len(clean_text) > 600 else "")
        try:
            self.tts_engine.say(short_text)
            self.tts_engine.runAndWait()
        except Exception as e:
            logging.error(f"Помилка відтворення аудіо: {e}")

    def listen(self) -> str:
        try:
            with sr.Microphone() as source:
                print("\n[Мікрофон увімкнено. Говоріть...]")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(
                    source,
                    timeout=Config.STT_TIMEOUT,
                    phrase_time_limit=Config.STT_PHRASE_LIMIT
                )
                print("[Розпізнавання...]")
                return self.recognizer.recognize_google(audio, language=Config.STT_LANGUAGE)
        except sr.WaitTimeoutError:
            logging.warning("Час очікування голосового вводу вичерпано.")
            return ""
        except sr.UnknownValueError:
            logging.warning("Не вдалося розпізнати мову.")
            return ""
        except Exception as e:
            logging.error(f"Помилка мікрофона: {e}")
            return ""


# Абстракція для мовних моделей
class LLMProvider:
    def generate_response(self, prompt: str, history: list) -> str:
        raise NotImplementedError


class CloudLLMProvider(LLMProvider):
    def __init__(self):
        try:
            from google import genai
            from google.genai import types
            import os

            api_key = os.environ.get("GEMINI_API_KEY")

            if not api_key:
                logging.error("API-ключ Gemini не знайдено.")
                self.is_ready = False
                return

            self.client = genai.Client(api_key=api_key)
            self.model_name = 'gemini-2.5-flash'
            self.types = types
            self.is_ready = True

        except Exception as e:
            logging.error(f"Помилка ініціалізації Gemini: {e}")
            self.is_ready = False

    def generate_response(self, user_input: str, history: list) -> str:
        if not self.is_ready:
            return "Помилка: Модель Gemini не ініціалізована. Перевірте API-ключ."

        gemini_contents = []

        # Конвертуємо нашу історію у формат types.Content для нового SDK
        for msg in history:
            role = "model" if msg["role"] == "assistant" else "user"
            gemini_contents.append(
                self.types.Content(
                    role=role,
                    parts=[self.types.Part.from_text(text=msg["content"])]
                )
            )

        # Додаємо поточне питання від користувача
        gemini_contents.append(
            self.types.Content(
                role="user",
                parts=[self.types.Part.from_text(text=user_input)]
            )
        )

        try:
            # Використовуємо новий метод generate_content
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=gemini_contents,
                config=self.types.GenerateContentConfig(
                    system_instruction=Config.SYSTEM_PROMPT
                )
            )
            return response.text
        except Exception as e:
            logging.error(f"Помилка API запиту Gemini: {e}")
            return "Виникла помилка при генерації відповіді."


# Провайдер для локальних моделей
class LocalLLMProvider(LLMProvider):

    def __init__(self, model_path: str):
        try:
            from llama_cpp import Llama
            self.llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_threads=12,
                verbose=False
            )
            self.is_ready = True
        except Exception as e:
            logging.error(f"Помилка завантаження локальної моделі: {e}")
            self.is_ready = False

    def generate_response(self, user_input: str, history: list) -> str:
        if not self.is_ready:
            return "Помилка: Локальна модель не ініціалізована."

        # Формування базового промпту для локальної моделі
        prompt = f"System: {Config.SYSTEM_PROMPT}\n"
        for msg in history[-3:]:
            prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
        prompt += f"User: {user_input}\nAssistant: "

        try:
            output = self.llm(
                prompt,
                max_tokens=500,
                temperature=0.3,
                stop=["User:", "\n\n"]
            )
            return output['choices'][0]['text'].strip()
        except Exception as e:
            logging.error(f"Помилка локальної генерації: {e}")
            return "Виникла помилка при обробці запиту локальною моделлю."


# Ядро застосунку
class NLPAssistantApp:
    def __init__(self, use_local_llm=False, local_model_path=""):
        self.audio = AudioProcessor()
        self.history = []
        self.session_start = datetime.now()

        # Ініціалізація провайдера
        if use_local_llm and local_model_path:
            print("[Інфо] Ініціалізація локальної моделі...")
            self.llm = LocalLLMProvider(local_model_path)
        else:
            print("[Інфо] Ініціалізація хмарного API...")
            self.llm = CloudLLMProvider()

        if not os.path.exists(Config.HISTORY_DIR):
            os.makedirs(Config.HISTORY_DIR)

    # Зберігає історію поточного діалогу у файл.
    def save_session(self):
        if not self.history:
            return

        filename = f"session_{self.session_start.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(Config.HISTORY_DIR, filename)

        data = {
            "session_date": self.session_start.isoformat(),
            "messages": self.history
        }

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"\n[Інфо] Сесію збережено: {filepath}")
        except Exception as e:
            logging.error(f"Помилка збереження історії: {e}")

    def run(self):
        print(" NLP-Асистент готовий до роботи.")
        print(" Введіть текст, або натисніть Enter для голосового вводу.")
        print(" Команди: 'q' або 'вихід' для завершення.")

        greeting = "Систему запущено. Я готовий відповідати на ваші запитання."
        self.audio.speak(greeting)

        try:
            while True:
                user_input = input("\nВи: ").strip()

                if user_input.lower() in ['q', 'exit', 'вихід', 'quit']:
                    print("Завершення роботи...")
                    self.audio.speak("Роботу завершено. До побачення.")
                    break

                if not user_input:
                    user_input = self.audio.listen()
                    if not user_input:
                        continue
                    print(f"Розпізнано текст: {user_input}")

                # Отримання відповіді від LLM
                response = self.llm.generate_response(user_input, self.history)

                # Оновлення історії діалогу
                self.history.append({"role": "user", "content": user_input})
                self.history.append({"role": "assistant", "content": response})

                # Вивід та озвучення
                print(f"\nАсистент:\n{response}")
                self.audio.speak(response)

        except KeyboardInterrupt:
            print("\nПримусове завершення роботи.")
        finally:
            self.save_session()


# Точка входу
if __name__ == "__main__":
    # Для використання локальної моделі потрібно замінити параметри:
    # app = NLPAssistantApp(use_local_llm=True, local_model_path="models/mistral-7b-instruct.Q4_K_M.gguf")
    app = NLPAssistantApp(use_local_llm=False)
    app.run()
