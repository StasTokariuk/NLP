import re, time, random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import spacy
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from curl_cffi import requests as curl_requests
from datetime import datetime

matplotlib.use('Agg')

nlp = spacy.load("uk_core_news_sm", disable=["parser", "ner"])

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment",
)

embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


# Українська Правда
def get_links_up():
    links = []
    try:
        res = curl_requests.get("https://www.pravda.com.ua/news/", impersonate="chrome110", timeout=15)
        soup = BeautifulSoup(res.text, 'html.parser')

        for item in soup.select('.article_news_list'):
            t = item.select_one('.article_time')
            a = item.select_one('.article_title a')
            if t and a:
                href = a['href']
                if not href.startswith('http'): href = 'https://www.pravda.com.ua' + href
                links.append((t.text.strip(), href, 'UP'))
    except Exception as e:
        print(f"Помилка збору лінків УП: {e}")
    return links


def get_article_text_up(url):
    try:
        time.sleep(random.uniform(0.5, 1.5))
        res = curl_requests.get(url, impersonate="chrome110", timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        container = soup.select_one('.post_news_text') or soup.select_one('.entry-content')
        if not container: return ""
        paragraphs = container.find_all(['p', 'h3'])
        return " ".join([p.text.strip() for p in paragraphs if len(p.text.strip()) > 25])
    except:
        return ""


# Укрінформ
def get_links_ukrinform():
    links = []
    try:
        res = curl_requests.get("https://www.ukrinform.ua/block-lastnews", impersonate="chrome110", timeout=15)
        soup = BeautifulSoup(res.text, 'html.parser')
        articles = soup.select('.restList article') or soup.find_all('article')

        for item in articles:
            link_tag = item.select_one('h2 a') or item.select_one('section a') or item.select_one('a')
            if link_tag and link_tag.has_attr('href'):
                href = link_tag['href']
                if not href.startswith('http'): href = 'https://www.ukrinform.ua' + href
                links.append(("12:00", href, 'UKRINFORM'))
    except Exception as e:
        print(f"Помилка збору лінків Укрінформ: {e}")
    return links


def get_article_text_ukrinform(url):
    try:
        time.sleep(random.uniform(0.5, 1.5))
        res = curl_requests.get(url, impersonate="chrome110", timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        container = soup.select_one('.newsText') or soup.select_one('.article__text')
        if not container: return ""
        return " ".join(p.text.strip() for p in container.find_all('p') if len(p.text.strip()) > 25)
    except:
        return ""


# PIK
def get_links_pik(date_path):
    links = []
    try:
        res = curl_requests.get(f"https://pik.net.ua/{date_path}/page/1/", impersonate="chrome110", timeout=15)
        soup = BeautifulSoup(res.text, 'html.parser')
        for art in soup.select('article.main__category-width'):
            link_tag = art.select_one('.main__category-width--content a')
            if link_tag and link_tag.has_attr('href'):
                links.append(("12:00", link_tag['href'], 'PIK'))
    except Exception as e:
        print(f"Помилка збору лінків PIK: {e}")
    return links


def get_article_text_pik(url):
    try:
        time.sleep(random.uniform(0.5, 1.5))
        res = curl_requests.get(url, impersonate="chrome110", timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        container = soup.select_one('.entry-content') or soup.select_one('.post-content')
        if not container: return ""
        return " ".join(p.text.strip() for p in container.find_all('p') if len(p.text.strip()) > 25)
    except:
        return ""


def main():
    print("\nСТАРТ ЗБОРУ НОВИН")
    data = []
    today = datetime.now()

    # 1. УП
    up_links = get_links_up()[:10]
    for _, url, src in up_links:
        text = get_article_text_up(url)
        if text: data.append({"Source": src, "Link": url, "Text": text})

    # 2. Укрінформ
    ukr_links = get_links_ukrinform()[:10]
    for _, url, src in ukr_links:
        text = get_article_text_ukrinform(url)
        if text: data.append({"Source": src, "Link": url, "Text": text})

    # 3. ПІК
    date_str_pik = today.strftime('%Y/%m/%d')
    pik_links = get_links_pik(date_str_pik)[:10]
    for _, url, src in pik_links:
        text = get_article_text_pik(url)
        if text: data.append({"Source": src, "Link": url, "Text": text})

    df = pd.DataFrame(data)
    if df.empty:
        print("Не вдалося зібрати новини.")
        return

    df.to_csv("news_dataset.csv", index=False, encoding="utf-8-sig")
    print(f"Зібрано {len(df)} новин з {df['Source'].nunique()} джерел. Збережено у 'news_dataset.csv'")

    def clean_lemmatize(text):
        text = re.sub(r'[^а-яіїєґa-z\s]', ' ', text.lower())
        return " ".join([t.lemma_ for t in nlp(text) if not t.is_stop and len(t.text) > 3])

    df['Clean_Text'] = df['Text'].apply(clean_lemmatize)

    source_texts = df.groupby('Source')['Clean_Text'].apply(' '.join)

    if len(source_texts) > 1:
        print("\nГенерація нейромережевих ембеддінгів для порівняння джерел...")
        neural_embeddings = embedder.encode(source_texts.tolist())

        similarity = cosine_similarity(neural_embeddings)
        sim_df = pd.DataFrame(similarity, index=source_texts.index, columns=source_texts.index)
        print("Матриця подібності джерел (Neural Embeddings):")
        print(sim_df.round(3))
    else:
        print("Зібрано лише 1 джерело, неможливо порівняти.")
        sim_df = None

    label_map = {'positive': 'Позитив', 'neutral': 'Нейтрал', 'negative': 'Негатив'}


    def get_sentiment(text):
        try:
            res = sentiment_analyzer(text[:512])[0]
            return label_map.get(res['label'].lower(), 'Нейтрал')
        except:
            return 'Нейтрал'

    print("\nАналіз тональності за допомогою моделі RoBERTa...")
    df['Sentiment'] = df['Text'].apply(get_sentiment)
    sentiment_dist = df.groupby(['Source', 'Sentiment']).size().unstack(fill_value=0)

    # Візуалізація результатів
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Аналіз новинних стрічок', fontsize=14)

    if sim_df is not None:
        cax = axes[0].imshow(sim_df.values, cmap='Blues', vmin=0, vmax=1)
        axes[0].set_xticks(range(len(sim_df.columns)))
        axes[0].set_yticks(range(len(sim_df.index)))
        axes[0].set_xticklabels(sim_df.columns)
        axes[0].set_yticklabels(sim_df.index)
        axes[0].set_title('Семантична подібність')
        fig.colorbar(cax, ax=axes[0])
        for i in range(len(sim_df)):
            for j in range(len(sim_df.columns)):
                axes[0].text(j, i, f"{sim_df.values[i, j]:.2f}", ha='center', va='center')
    else:
        axes[0].set_title('Подібність контенту (Недостатньо даних)')

    colors = ['#4CAF50', '#9E9E9E', '#F44336']
    for col in ['Позитив', 'Нейтрал', 'Негатив']:
        if col not in sentiment_dist.columns:
            sentiment_dist[col] = 0
    sentiment_dist = sentiment_dist[['Позитив', 'Нейтрал', 'Негатив']]

    sentiment_dist.plot(kind='bar', stacked=True, ax=axes[1], color=colors)
    axes[1].set_title('Тональність новин по джерелах')
    axes[1].set_ylabel('Кількість новин')
    axes[1].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig('lab6_results.png', dpi=150)


if __name__ == "__main__":
    main()