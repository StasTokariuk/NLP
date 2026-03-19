import requests, re, time, random, os
from bs4 import BeautifulSoup
import pandas as pd
from collections import Counter
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib
import ast
import spacy

matplotlib.use('TkAgg')


# Стеммер на основі алгоритму Портера для української мови
class UkrainianStemmer:
    def __init__(self):
        self.vowels = r'аеиоуюяіїє'
        self.perfectiveground = r'(ив|ивши|ившись|ав|авши|авшись)$'
        self.reflexive = r'(с[яь])$'
        self.adjective = r'(ий|ого|ому|им|ім|а|ої|ою|е|і|их|ими|їй|ї|я|єю|є)$'
        self.participle = r'(ний|ного|ному|ним|нім|на|ної|ною|не|ні|них|ними|тий|того|тому|тим|тім|та|тої|тою|те|ті|тих|тими)$'
        self.verb = r'(ти|ть|сь|ся|ив|ав|яв|у|ю|а|я|е|є|и|і|ї|й|ло|ла|ли|но|то)$'
        self.noun = r'(а|я|о|е|и|і|ї|й|ю|ям|ями|ях|ем|ом|ою|ею|єю|ів|їв|ам|ами|ах|у|ові|еві|єві|ей|єю)$'
        self.derivational = r'[^аеиоуюяіїє][аеиоуюяіїє]+[^аеиоуюяіїє]+[аеиоуюяіїє].*(ість|ність)$'

    def stem(self, word):
        word = word.lower()
        if not re.search(self.vowels, word): return word
        word = re.sub(self.reflexive, '', word)
        if re.search(self.adjective, word):
            word = re.sub(self.adjective, '', word)
        elif re.search(self.participle, word):
            word = re.sub(self.participle, '', word)
        else:
            if not re.sub(self.verb, '', word) == word:
                word = re.sub(self.verb, '', word)
            else:
                word = re.sub(self.noun, '', word)
        word = re.sub(r'и$', '', word)
        if re.search(self.derivational, word): word = re.sub(r'(ість|ність)$', '', word)
        word = re.sub(r'ь$', '', word)
        return word


HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
RESULTS_DIR = 'nlp_results'
FILE_STATS = f'{RESULTS_DIR}/content_stats.csv'
FILE_RAW_CSV = f'{RESULTS_DIR}/raw_articles.csv'

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Ініціалізація NLP-моделей
nlp = spacy.load("uk_core_news_sm")
custom_stop_words = {'березень', 'український', 'російський', 'повідомити', 'новий', 'деталь'}
for word in custom_stop_words:
    nlp.vocab[word].is_stop = True
stemmer = UkrainianStemmer()


def get_article_text(url):
    try:
        time.sleep(random.uniform(0.5, 1.5))
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        container = soup.select_one('.post_news_text') or soup.select_one('.entry-content')
        if not container: return ""
        paragraphs = container.find_all(['p', 'h3'])
        return " ".join([p.text.strip() for p in paragraphs if len(p.text) > 25])
    except:
        return ""


def get_links_up(date_str):
    links = []
    try:
        soup = BeautifulSoup(requests.get(f"https://www.pravda.com.ua/news/date_{date_str}/", headers=HEADERS).text,
                             'html.parser')
        for item in soup.select('.article_news_list'):
            t, a = item.select_one('.article_time'), item.select_one('.article_title a')
            if t and a: links.append((t.text, a['href'], 'UP'))
    except:
        pass
    return links


def get_links_pik(date_path):
    links = []
    for page in range(1, 4):
        try:
            soup = BeautifulSoup(requests.get(f"https://pik.net.ua/{date_path}/page/{page}/", headers=HEADERS).text, 'html.parser')
            for art in soup.select('article.main__category-width'):
                time_box = art.select_one('.date')
                link_tag = art.select_one('.main__category-width--content a')
                if time_box and link_tag:
                    match = re.search(r'(\d{1,2}:\d{2})', time_box.text)
                    links.append((match.group(1) if match else "12:00", link_tag['href'], 'PIK'))
        except:
            continue
    return links

# NLP: нормалізація, токенізація, фільтрація, лематизація, стемінг
def process_nlp_pipeline(texts, day_marker):
    if not texts: return "", [], 0

    raw_text = " ".join(texts)
    clean_text = re.sub(r'\s+', ' ', re.sub(r'[^а-яіїєґa-z\s]', ' ', raw_text.lower())).strip()

    doc = nlp(clean_text)
    tokens = [token.text for token in doc]
    processed_lemmas, processed_stems, no_stopwords_tokens = [], [], []

    for token in doc:
        # Фільтрація за стоп-словами та довжиною
        if not token.is_stop and token.is_alpha and len(token.text) > 3 and token.lemma_ not in custom_stop_words:
            no_stopwords_tokens.append(token.text)
            processed_lemmas.append(token.lemma_)
            processed_stems.append(stemmer.stem(token.lemma_))

    counts = Counter(processed_lemmas).most_common(10)

    # Збереження проміжних результатів
    with open(f'{RESULTS_DIR}/1_raw_texts.txt', 'a', encoding='utf-8') as f:
        f.write(f"\n--- {day_marker} ---\n{raw_text[:1000]}...\n")
    with open(f'{RESULTS_DIR}/2_filtered.txt', 'a', encoding='utf-8') as f:
        f.write(f"\n--- {day_marker} ---\n{clean_text[:1000]}...\n")
    with open(f'{RESULTS_DIR}/3_tokens.txt', 'a', encoding='utf-8') as f:
        f.write(f"\n--- {day_marker} ---\n{', '.join(tokens[:200])}\n")
    with open(f'{RESULTS_DIR}/4_no_stopwords.txt', 'a', encoding='utf-8') as f:
        f.write(f"\n--- {day_marker} ---\n{', '.join(no_stopwords_tokens[:200])}\n")
    with open(f'{RESULTS_DIR}/5_lemmatized.txt', 'a', encoding='utf-8') as f:
        f.write(f"\n--- {day_marker} ---\n{', '.join(processed_lemmas[:200])}\n")
    with open(f'{RESULTS_DIR}/6_stemmed.txt', 'a', encoding='utf-8') as f:
        f.write(f"\n--- {day_marker} ---\n{', '.join(processed_stems[:200])}\n")

    if not counts: return "", [], 0
    return ", ".join([c[0] for c in counts]), [c[1] for c in counts], sum([c[1] for c in counts])


# Збір даних за 14 днів
if not os.path.exists(FILE_STATS):
    all_data, raw_data = [], []
    today = datetime.now()

    for i in range(14, 0, -1):
        curr_date = today - timedelta(days=i)
        d_display = curr_date.strftime('%d.%m.%Y')
        print(f"\nОБРОБКА ДАТИ: {d_display} (Залишилось днів: {i})")

        day_links = get_links_up(curr_date.strftime('%d%m%Y')) + get_links_pik(curr_date.strftime('%Y/%m/%d'))
        random.shuffle(day_links)
        day_links = day_links[:30]

        windows = {'Ранок': [], 'Обід': [], 'Вечір': []}
        for t_str, url, src in day_links:
            hour = int(re.search(r'^(\d+)', t_str).group(1))
            cat = 'Ранок' if hour <= 10 else ('Обід' if hour <= 17 else 'Вечір')
            windows[cat].append((url, src))

        for cat, links in windows.items():
            if not links: continue
            period_texts = []
            for url, src in links:
                text = get_article_text(url)
                if text:
                    period_texts.append(text)
                    raw_data.append([d_display, cat, src, url, text])

            top_w, freqs, total = process_nlp_pipeline(period_texts, f"{d_display} [{cat}]")
            if total > 0: all_data.append([d_display, cat, top_w, str(freqs), total])

        pd.DataFrame(all_data, columns=['День', 'Час', 'Топ 10', 'Частота', 'Сума']).to_csv(FILE_STATS, index=False, encoding='utf-8-sig')
        pd.DataFrame(raw_data, columns=['День', 'Час', 'Джерело', 'Посилання', 'Текст']).to_csv(FILE_RAW_CSV, index=False, encoding='utf-8-sig')
        if i > 1: time.sleep(5)

# Візуалізація результатів
print("\nБудуємо графіки")
df = pd.read_csv(FILE_STATS)
word_trends = {}

for _, row in df.iterrows():
    if pd.isna(row['Топ 10']) or pd.isna(row['Частота']): continue
    words = [w.strip() for w in str(row['Топ 10']).split(',')]
    freqs = ast.literal_eval(str(row['Частота']))
    day = row['День']

    for w, f in zip(words, freqs):
        if w not in word_trends: word_trends[w] = {}
        word_trends[w][day] = word_trends[w].get(day, 0) + f

top_global = sorted([w for w in word_trends.keys()], key=lambda w: sum(word_trends[w].values()), reverse=True)[:5]

plt.figure(figsize=(16, 7))
plt.subplot(1, 2, 1)
days_list = df['День'].unique()
for word in top_global:
    plt.plot(days_list, [word_trends[word].get(d, 0) for d in days_list], marker='o', linewidth=2, label=word)

plt.title('Трансформація новин (14 днів)')
plt.xticks(rotation=45)
plt.ylabel('Частота згадувань')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
all_words_text = " ".join(df['Топ 10'].dropna().astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(all_words_text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Домінантні теми періоду')

plt.tight_layout()
plt.show()