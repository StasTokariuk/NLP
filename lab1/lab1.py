import requests, re, time, random, os
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib

matplotlib.use('TkAgg')

# конфіги
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
FILE_NAME = 'content.csv'
STOP_WORDS = {'та', 'і', 'в', 'на', 'що', 'це', 'як', 'за', 'до', 'про', 'для', 'не', 'по', 'з', 'але', 'все', 'тільки',
              'його', 'також', 'від', 'було', 'яких', 'ще', 'вже', 'коли', 'через', 'після', 'який', 'того', 'цього',
              'березня', 'понад', 'проти', 'деталі', 'джерело', 'заявив', 'станом', 'неділю', 'вівторок', 'словами',
              'імені', 'зокрема', 'можуть', 'щодо', 'якщо', 'серед', 'більше', 'однак', 'лише', 'внаслідок'}


# функції

# Витягує текст статті
def get_article_text(url):
    try:
        time.sleep(random.uniform(0.8, 2.8))  # Для запобігання бану
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')

        # список контейнерів для УП, ПІК
        classes = ['.post_news_text', '.entry-content']
        container = None
        for c in classes:
            container = soup.select_one(c)
            if container: break

        if not container: return ""
        paragraphs = container.find_all(['p', 'h3'])
        return " ".join([p.text.strip() for p in paragraphs if len(p.text) > 25])
    except:
        return ""

# збираємо посилання та час з УП
def get_links_up(date_str):
    links = []
    url = f"https://www.pravda.com.ua/news/date_{date_str}/"
    try:
        soup = BeautifulSoup(requests.get(url, headers=HEADERS).text, 'html.parser')
        for item in soup.select('.article_news_list'):
            t, a = item.select_one('.article_time'), item.select_one('.article_title a')
            if t and a: links.append((t.text, a['href'], 'UP'))
    except:
        pass
    return links

# збираємо посилання та час з ПІК
def get_links_pik(date_path):
    links = []
    for page in range(1, 4):
        try:
            soup = BeautifulSoup(requests.get(f"https://pik.net.ua/{date_path}/page/{page}/", headers=HEADERS).text,'html.parser')
            for art in soup.select('article.main__category-width'):
                time_box = art.select_one('.date')
                link_tag = art.select_one('.main__category-width--content a')
                if time_box and link_tag:
                    match = re.search(r'(\d{1,2}:\d{2})', time_box.text)
                    links.append((match.group(1) if match else "12:00", link_tag['href'], 'PIK'))
        except:
            continue
    return links

# шукаємо топ 5 слів
def process_nlp(texts):
    words = re.findall(r'\b[а-яіїєґ]{4,}\b', " ".join(texts).lower())
    counts = Counter([w for w in words if w not in STOP_WORDS]).most_common(5)
    return ", ".join([c[0] for c in counts]), [c[1] for c in counts], sum([c[1] for c in counts])


# збір даних

if not os.path.exists(FILE_NAME):
    all_data = []
    today = datetime(2026, 3, 10)

    for i in range(7, 0, -1):
        curr_date = today - timedelta(days=i)
        d_display = curr_date.strftime('%d.%m.%Y')
        print(f"\nОбробка дати: {d_display}")

        day_links = get_links_up(curr_date.strftime('%d%m%Y')) + get_links_pik(curr_date.strftime('%Y/%m/%d'))
        random.shuffle(day_links)
        print(f"    Знайдено статей: {len(day_links)}")

        windows = {'Ранок': [], 'Обід': [], 'Вечір': []}
        for t_str, url, src in day_links:
            hour = int(re.search(r'^(\d+)', t_str).group(1))
            cat = 'Ранок' if hour <= 10 else ('Обід' if hour <= 17 else 'Вечір')
            windows[cat].append((url, src))

        for cat, links in windows.items():
            if not links: continue
            print(f"    {cat}: парсимо {len(links)} статей ")

            period_texts = [get_article_text(url) for url, _ in links]
            top_w, freqs, total = process_nlp([t for t in period_texts if t])
            all_data.append([d_display, cat, top_w, freqs, total])

        # зберігаємо ітеративно
        pd.DataFrame(all_data, columns=['День', 'Час', 'Топ 5', 'Частота', 'Сума']).to_csv(FILE_NAME, index=False, encoding='utf-8-sig')
        print("    Пауза 5 хв")
        if i > 1: time.sleep(300)  # пауза 5 хв між днями

# аналіз

df = pd.read_csv(FILE_NAME)
y = df['Сума'].values
x = np.arange(len(y))

# математична модель прямої
a, b = np.polyfit(x, y, 1)
forecast_x = np.arange(len(x) + 6)
forecast_y = a * forecast_x + b

plt.figure(figsize=(15, 7))

# графік динаміки
plt.subplot(1, 2, 1)
plt.plot(x, y, 'b-o', label='Фактична частота')
plt.plot(forecast_x, forecast_y, 'r--', label='МНК')
plt.title('Інтенсивність новин');
plt.legend();
plt.grid(True, alpha=0.3)

# хмара слів
plt.subplot(1, 2, 2)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(df['Топ 5']))
plt.imshow(wordcloud, interpolation='bilinear');
plt.axis('off')
plt.title('Ключові теми')

plt.tight_layout();
plt.show()