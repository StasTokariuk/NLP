from curl_cffi import requests as curl_requests
from bs4 import BeautifulSoup
import time
import random
import pandas as pd
from datetime import datetime, timedelta


# Змінили функцію для роботи з датою
def get_links_up(date_str):
    links = []
    try:
        # Формуємо URL за датою
        url = f"https://www.pravda.com.ua/news/date_{date_str}/"
        res = curl_requests.get(url, impersonate="chrome110", timeout=15)
        soup = BeautifulSoup(res.text, 'html.parser')

        for item in soup.select('.article_news_list'):
            t = item.select_one('.article_time')
            a = item.select_one('.article_title a')

            if t and a:
                href = a['href']
                if not href.startswith('http'):
                    href = 'https://www.pravda.com.ua' + href

                # Фільтруємо піддомени (epravda, eurointegration тощо)
                if 'www.pravda.com.ua' in href:
                    links.append((t.text.strip(), href, 'UP'))
    except Exception as e:
        print(f"Помилка збору лінків за {date_str}: {e}")
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


def main():
    target_valid_count = 70
    news_data = []
    seen_links = set()

    # Беремо поточну дату для старту
    current_date = datetime.now()

    print(f"Починаємо збір {target_valid_count} повноцінних новин, починаючи з {current_date.strftime('%d.%m.%Y')}...")

    # Цикл працює, поки не зберемо 70 текстів
    while len(news_data) < target_valid_count:
        # Формат дати на УП - ddmmyyyy (наприклад: 30042026)
        date_str = current_date.strftime("%d%m%Y")

        print(f"\n--- Скануємо новини за {current_date.strftime('%d-%m-%Y')} ---")
        new_links = get_links_up(date_str)

        if not new_links:
            print(f"За {date_str} новин не знайдено. Переходимо до попереднього дня.")
        else:
            for time_str, url, source in new_links:
                if len(news_data) >= target_valid_count:
                    break

                if url in seen_links:
                    continue
                seen_links.add(url)

                text = get_article_text_up(url)

                if text:
                    news_data.append({
                        'Date': current_date.strftime("%Y-%m-%d"),  # Додали колонку з датою
                        'Time': time_str,
                        'Source': source,
                        'Link': url,
                        'Text': text
                    })
                    print(f"[{len(news_data)}/{target_valid_count}] Успішно: {url}")
                else:
                    print(f"[ПРОПУСК] Немає тексту: {url}")

        # Віднімаємо 1 день для наступної ітерації
        current_date -= timedelta(days=1)
        time.sleep(random.uniform(1, 2))

    # Збереження результатів
    df = pd.DataFrame(news_data)
    filename = "up_news.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\nГотово! Збережено рівно {len(df)} новин у файл {filename}")


if __name__ == "__main__":
    main()