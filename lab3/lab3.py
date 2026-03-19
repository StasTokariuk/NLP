import requests
from bs4 import BeautifulSoup
import spacy
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import re

import matplotlib
matplotlib.use('TkAgg')

# Ініціалізація NLP-моделі для української мови
try:
    nlp = spacy.load("uk_core_news_sm")
except OSError:
    print("Завантажте мовну модель: python -m spacy download uk_core_news_sm")
    exit()

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

# Збирає тексти останніх новин з Української Правди.
def get_news_texts(limit=15):
    texts = []
    url = "https://www.pravda.com.ua/news/"

    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')

        links = [a['href'] for a in soup.select('.article_news_list .article_title a')][:limit]

        for link in links:
            if not link.startswith('http'):
                link = "https://www.pravda.com.ua" + link

            article_res = requests.get(link, headers=HEADERS, timeout=10)
            article_soup = BeautifulSoup(article_res.text, 'html.parser')
            container = article_soup.select_one('.post_news_text')

            if container:
                raw_text = " ".join([p.text.strip() for p in container.find_all('p')])
                clean_text = re.sub(r'\s+', ' ', re.sub(r'[^а-яіїєґА-ЯІЇЄҐa-zA-Z\s]', ' ', raw_text)).strip()

                if len(clean_text) > 100:
                    texts.append(clean_text.lower())
    except Exception as e:
        print(f"Помилка під час парсингу: {e}")

    return texts

# Виконує POS-тегування та повертає частоту кожної частини мови.
def pos_tagging_analysis(texts):
    all_pos = []
    for text in texts:
        doc = nlp(text)
        for token in doc:
            if token.is_alpha and not token.is_stop:
                all_pos.append(token.pos_)
    return Counter(all_pos)

# Застосовує векторизацію (TF-IDF) для визначення тематичної ваги слів.
def vectorization_analysis(texts):
    lemmatized_texts = [" ".join([token.lemma_ for token in nlp(t) if token.is_alpha and not token.is_stop]) for t in texts]

    vectorizer = TfidfVectorizer(max_features=10)
    tfidf_matrix = vectorizer.fit_transform(lemmatized_texts)

    feature_names = vectorizer.get_feature_names_out()
    sums = tfidf_matrix.sum(axis=0).A1

    words_freq = [(word, sums[idx]) for word, idx in zip(feature_names, range(len(sums)))]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq


if __name__ == "__main__":
    print("Збір свіжих новин зі стрічки")
    news_texts = get_news_texts(limit=15)

    if not news_texts:
        print("Не вдалося зібрати новини.")
    else:
        print(f"Успішно зібрано {len(news_texts)} статей.")

        # Аналіз тегів (POS Tagging)
        pos_counts = pos_tagging_analysis(news_texts)
        pos_labels, pos_values = zip(*pos_counts.most_common())

        # Векторизація тексту
        top_vectors = vectorization_analysis(news_texts)
        vec_labels, vec_values = zip(*top_vectors)

        # Графічна візуалізація результатів
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        plt.bar(pos_labels, pos_values, color='#4C72B0')
        plt.title('Частотний аналіз частин мови (POS Tags)')
        plt.xlabel('Частина мови (Spacy POS)')
        plt.ylabel('Кількість входжень')

        for i, v in enumerate(pos_values):
            plt.text(i, v + 2, str(v), ha='center', va='bottom', fontsize=9)

        plt.subplot(1, 2, 2)
        plt.bar(vec_labels, vec_values, color='#55A868')
        plt.title('Топ-10 слів за векторною вагою (TF-IDF)')
        plt.xlabel('Лема')
        plt.ylabel('Сумарна вага в текстах')
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.show()