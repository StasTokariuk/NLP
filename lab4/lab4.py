import requests
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from nltk.util import ngrams
from nltk.text import Text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score

import matplotlib
matplotlib.use('TkAgg')

nlp = spacy.load("uk_core_news_sm")


HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
RANDOM_SEED = 42

# WEB-СКРАПІНГ ТА ПІДГОТОВКА ДАНИХ

def get_news(limit=80):
    texts = []
    url = "https://www.pravda.com.ua/news/"
    res = requests.get(url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(res.text, 'html.parser')
    links = [a['href'] for a in soup.select('.article_news_list .article_title a')][:limit]

    for link in links:
        full_url = link if link.startswith('http') else "https://www.pravda.com.ua" + link
        try:
            art_res = requests.get(full_url, headers=HEADERS, timeout=10)
            art_soup = BeautifulSoup(art_res.text, 'html.parser')
            container = art_soup.select_one('.post_news_text')
            if container:
                txt = " ".join([p.text.strip() for p in container.find_all('p')])
                if len(txt) > 100: texts.append(txt)
        except: continue
    return texts

def preprocess(text):
    doc = nlp(text.lower())
    return " ".join([t.lemma_ for t in doc if not t.is_stop and not t.is_punct and t.is_alpha and len(t.lemma_) > 1])

print("1. Збір даних та синтез фейків")
real_raw = get_news(limit=45)
fake_raw = [
    "на марсі знайшли поклади українського сала та підземні вишневі садки",
    "уряд ухвалив закон про обов'язкове безкоштовне роздавання криптокотів усім студентам",
    "вчені довели що вивчення nlp замінює вісім годин сну та чашку кави",
    "завтра всі іспити скасовано через візит інопланетян до кпі імені ігоря сікорського",
    "білл гейтс купує кожен університет в україні щоб роздати безкоштовні чіпси"
]

all_raw = real_raw + fake_raw
labels = [0] * len(real_raw) + [1] * len(fake_raw)
all_cleaned = [preprocess(t) for t in all_raw]

# ЧАСТОТНИЙ ТА ЙМОВІРНІСНИЙ АНАЛІЗ

# 1. TF-IDF Векторизація
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(all_cleaned)

# 2. Біграмний аналіз
all_words = " ".join(all_cleaned).split()
bigrams = list(ngrams(all_words, 2))
print("\nтоп-5 біграм:", pd.Series(bigrams).value_counts().head(5))

# 3. Розподіл довжини слів
plt.figure(figsize=(8, 4))
plt.hist([len(w) for w in all_words], bins=15, color='skyblue', edgecolor='black')
plt.title("Розподіл довжини слів")
plt.show()

# 4. Лексична дисперсія
print("\nПобудова графіка дисперсії")
tokens_list = Text(all_words)
words_to_plot = ["марс", "сало", "зеленський", "трамп", "війна"]

tokens_list.dispersion_plot(words_to_plot)
plt.show()

# КЛАСТЕРИЗАЦІЯ ТА ВЕРИФІКАЦІЯ

# Класифікація
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=RANDOM_SEED)
clf = RandomForestClassifier(random_state=RANDOM_SEED).fit(X_train, y_train)
print(f"\nТочність класифікації: {accuracy_score(y_test, clf.predict(X_test)):.2%}")

# Кластеризація без вчителя
kmeans = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=10).fit(X)
clusters = kmeans.labels_

# Оцінка
ari = adjusted_rand_score(labels, clusters)
print(f"Ефективність кластеризації (ARI): {ari:.4f}")

# Фінальна таблиця
df = pd.DataFrame({'Текст': [t[:50] for t in all_raw], 'Мітка': labels, 'Кластер': clusters})
print("\nРезультати (фрагмент):")
print(df.tail(10))