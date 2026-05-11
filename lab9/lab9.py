from __future__ import annotations
import re
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field
from collections import Counter

import pymorphy3

# CONFIG

LANG = "uk"
MODE = "text"

LABELS = {
    "uk": {
        "welcome": "Вiтаємо у BudMarket Plus!",
        "ask_name": "Як вас звати? ",
        "greet": "Вiтаємо, {}!",
        "prompt": "Запит: ",
        "bye": "До побачення!",
        "not_found": "Не знайдено.",
        "no_results": "Нiчого не знайдено: {}",
        "found": "Знайдено ({}):",
        "reviews_hdr": "Вiдгуки: {}",
        "sentiment": "Тональнiсть (BoW): позит={} нейтр={} негат={} score={:+.2f}",
        "compare_hdr": "Порiвняння: {} vs {}",
        "cheaper": "Дешевший: {}",
        "better_rev": "Кращi вiдгуки: {}",
        "cos_sim": "Косинусна подiбнiсть TF-IDF: {:.3f}",
        "nlp_hdr": "NLP-конвеєр: {}",
        "dept_hdr": "Вiддiл {}: {} | {}",
        "lang_switch": "Мова: Українська",
        "mode_switch": "Режим: {}",
        "help": (
            "Команди:\n"
            "  вiддiли                   - список вiддiлiв\n"
            "  вiддiл <N>                - товари вiддiлу\n"
            "  товар <назва|КОД>         - пошук / картка\n"
            "  вiдгуки <КОД>             - вiдгуки + тональнiсть\n"
            "  порiвняти <КОД> <КОД>     - порiвняння\n"
            "  nlp <текст>               - NLP-конвеєр\n"
            "  мова / language           - змiнити мову\n"
            "  режим / mode              - змiнити режим\n"
            "  вихiд / exit              - завершити"
        ),
    },
    "en": {
        "welcome": "Welcome to BudMarket Plus!",
        "ask_name": "What is your name? ",
        "greet": "Welcome, {}!",
        "prompt": "Query: ",
        "bye": "Goodbye!",
        "not_found": "Not found.",
        "no_results": "Nothing found for: {}",
        "found": "Found ({}):",
        "reviews_hdr": "Reviews: {}",
        "sentiment": "Sentiment (BoW): pos={} neu={} neg={} score={:+.2f}",
        "compare_hdr": "Compare: {} vs {}",
        "cheaper": "Cheaper: {}",
        "better_rev": "Better reviews: {}",
        "cos_sim": "Cosine similarity TF-IDF: {:.3f}",
        "nlp_hdr": "NLP pipeline: {}",
        "dept_hdr": "Dept {}: {} | {}",
        "lang_switch": "Language: English",
        "mode_switch": "Mode: {}",
        "help": (
            "Commands:\n"
            "  departments               - list departments\n"
            "  department <N>            - dept products\n"
            "  product <name|CODE>       - search / card\n"
            "  reviews <CODE>            - reviews + sentiment\n"
            "  compare <CODE> <CODE>     - comparison\n"
            "  nlp <text>                - NLP pipeline\n"
            "  language / мова           - switch language\n"
            "  mode / режим              - switch mode\n"
            "  exit / вихiд              - quit"
        ),
    },
}


def L(key: str, *args) -> str:
    s = LABELS[LANG][key]
    if args:
        return s.format(*args)
    return s


# NLP

STOPWORDS = {
    "uk": frozenset({"i", "й", "та", "що", "як", "але", "або", "не", "вже", "ще", "у", "в", "на", "за",
                     "по", "до", "вiд", "з", "iз", "про", "при", "мiж", "через", "для", "без", "пiсля",
                     "вiн", "вона", "вони", "ми", "ви", "я", "ти", "його", "їх", "той", "тi", "цей", "це",
                     "так", "де", "коли", "хто", "був", "була", "були", "буде", "є"}),
    "en": frozenset({"a", "an", "the", "and", "or", "of", "in", "for", "to", "is", "are", "was",
                     "were", "be", "been", "it", "its", "this", "that", "these", "those", "i", "you",
                     "he", "she", "we", "they", "at", "by", "from", "with", "on", "as", "not"}),
}

_MORPH = pymorphy3.MorphAnalyzer(lang="uk")

_POS_MAP = {
    "NOUN": "noun", "ADJF": "adj", "ADJS": "adj", "VERB": "verb",
    "INFN": "verb", "ADVB": "adv", "NUMR": "num", "NPRO": "pron",
    "PREP": "prep", "CONJ": "conj", "PRCL": "part"
}

_SUFFIXES = (
    "уватися", "ватися", "тися", "увати", "ювати", "вати", "ати", "яти", "ити", "iти",
    "ськiсть", "iсть", "ання", "iння", "ський", "овий", "євий", "ових", "овi",
    "ними", "ного", "ному", "нiй", "ами", "ями", "ого", "ому", "ою", "iй", "ий",
    "iв", "ях", "ах", "и", "i", "а", "я", "е", "у", "ю"
)


def stem(word: str) -> str:
    word = word.lower()
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zа-яiїєґ]+", text.lower())


def normalize(text: str) -> list[str]:
    sw = STOPWORDS["uk"] | STOPWORDS["en"]
    tokens = tokenize(text)
    normalized = []
    for token in tokens:
        if token not in sw:
            normalized.append(stem(token))
    return normalized


def pos_tag(tokens: list[str]) -> list[tuple[str, str, str]]:
    result = []
    for token in tokens:
        parsed = _MORPH.parse(token)[0]
        pos = _POS_MAP.get(parsed.tag.POS or "", "x")
        result.append((token, pos, parsed.normal_form))
    return result


def tfidf(query: list[str], doc: list[str], idf: dict[str, float]) -> float:
    tf = Counter(doc)
    n = len(doc) or 1
    score = 0.0
    for s in query:
        score += (tf[s] / n) * idf.get(s, 0)
    return score


# DATA

@dataclass(frozen=True)
class Product:
    id: str
    name_uk: str
    name_en: str
    unit: str
    price: int
    stock: int
    desc_uk: str
    desc_en: str
    reviews: tuple[str, ...] = field(default_factory=tuple)

    @property
    def name(self) -> str:
        return self.name_uk if LANG == "uk" else self.name_en

    @property
    def desc(self) -> str:
        return self.desc_uk if LANG == "uk" else self.desc_en

    @property
    def text(self) -> str:
        return f"{self.name} {self.desc}"


@dataclass(frozen=True)
class Department:
    id: str
    name_uk: str
    name_en: str
    zone: str
    desc_uk: str
    desc_en: str
    products: tuple[Product, ...]

    @property
    def name(self) -> str:
        return self.name_uk if LANG == "uk" else self.name_en

    @property
    def desc(self) -> str:
        return self.desc_uk if LANG == "uk" else self.desc_en


CATALOG: tuple[Department, ...] = (
    Department("1", "Будiвельнi матерiали", "Building Materials", "Zone A / Floor 1",
               "Цемент, блоки, цегла, арматура", "Cement, blocks, brick, rebar", (
                   Product("A01", "Цемент М500", "Cement M500", "bag 25kg", 185, 240,
                           "Портландцемент М500, для фундаментiв та стяжок.",
                           "Portland cement M500, for foundations and screeds.",
                           ("Вiдмiнна якiсть!", "Схопився швидко.", "Мiшок рваний був.")),
                   Product("A02", "Газобетонний блок D500", "Aerated block D500", "pcs", 52, 1800,
                           "600x200x300 мм. Легкий, хороша теплоiзоляцiя.",
                           "600x200x300mm. Light, good thermal insulation.",
                           ("Рiвнi блоки.", "Задоволений покупкою.", "Крихкуватi.")),
                   Product("A03", "Арматура А3 12 мм", "Rebar A3 12mm", "lin.m", 38, 5000,
                           "Ребриста, сталь Ст3, для залiзобетонних конструкцiй.",
                           "Ribbed, St3 steel, for reinforced concrete structures.",
                           ("Гарна якiсть металу.", "Все чiтко.", "Рекомендую.")),
               )),
    Department("2", "Оздоблювальнi матерiали", "Finishing Materials", "Zone B / Floor 1",
               "Штукатурки, фарби, плитка", "Plasters, paints, tiles", (
                   Product("B01", "Шпаклiвка Knauf", "Knauf Filler", "bag 25kg", 320, 150,
                           "Гiпсова фiнiшна шпаклiвка для стiн i стель.",
                           "Gypsum finish filler for walls and ceilings.",
                           ("Iдеально рiвна поверхня!", "Витрата менша.", "Трохи дорога.")),
                   Product("B02", "Фарба iнтер'єрна 10 л", "Interior paint 10L", "bucket", 650, 80,
                           "Акрилова, миюча, матова. Витрата 8-10 м2/л.",
                           "Acrylic, washable, matte. Coverage 8-10m2/L.",
                           ("Добре покриває.", "Запах невеликий.", "Рекомендую.")),
                   Product("B03", "Плитка керамiчна 30x60", "Ceramic tile 30x60", "m2", 420, 500,
                           "Глазурована, для ванних кiмнат та кухонь.",
                           "Glazed, for bathrooms and kitchens.",
                           ("Гарний малюнок.", "Вiдкол на однiй плитцi.", "Чудово.")),
               )),
    Department("3", "Iнструменти", "Tools & Fasteners", "Zone C / Floor 2",
               "Електроiнструмент, дюбелi, анкери", "Power tools, plugs, anchors", (
                   Product("C01", "Перфоратор Bosch GBH 2-26", "Bosch GBH 2-26 Hammer", "pcs", 4800, 12,
                           "800 Вт, SDS-plus, 2.7 Дж, свердлiння/удар/долото.",
                           "800W, SDS-plus, 2.7J, drilling/hammer/chisel.",
                           ("Потужний!", "Бере бетон без проблем.", "Шумнуватий.")),
                   Product("C02", "Анкер-болт M10x100", "Anchor bolt M10x100", "pack 25", 145, 300,
                           "Розпiрний, для бетону та цегли, оцинкований.",
                           "Expansion anchor for concrete and brick, zinc plated.",
                           ("Тримає надiйно.", "Все ок.")),
                   Product("C03", "Рiвень лазерний 3D", "3D Laser Level", "pcs", 1250, 8,
                           "3 лiнiї, точнiсть 0.3 мм/м, дальнiсть 15 м.",
                           "3 lines, accuracy 0.3mm/m, range 15m.",
                           ("Дуже зручний!", "Точний.", "Акумулятор сiдає швидко.")),
               )),
    Department("4", "Електрика та сантехнiка", "Electrical & Plumbing", "Zone D / Floor 2",
               "Кабель, труби, змiшувачi", "Cable, pipes, mixers", (
                   Product("D01", "Кабель ВВГнг 3x2.5", "Cable VVGng 3x2.5", "lin.m", 48, 2000,
                           "Силовий, вогнестiйка оболонка, внутрiшня проводка.",
                           "Power cable, fire-resistant sheath, indoor wiring.",
                           ("Стандарт!", "Якiсть хороша.", "Беру регулярно.")),
                   Product("D02", "Труба ПП 20 мм", "PP Pipe 20mm", "lin.m", 32, 1500,
                           "PP-R PN20, гаряче та холодне водопостачання.",
                           "PP-R PN20, hot and cold water supply.",
                           ("Не тече.", "Зручно паяти.", "Рекомендую.")),
                   Product("D03", "Змiшувач Grohe", "Grohe Mixer", "pcs", 3200, 5,
                           "Хромований, одноважiльний, з душовим гарнiтуром.",
                           "Chrome, single-lever, with shower set.",
                           ("Якiсть Grohe!", "Вiдмiнна цiна.", "Задоволений.")),
               )),
)


def _build_idf() -> dict[str, float]:
    docs = []
    for d in CATALOG:
        for p in d.products:
            docs.append(normalize(p.text))

    n_docs = len(docs)
    df = Counter()
    for doc in docs:
        for word in set(doc):
            df[word] += 1

    idf = {}
    for word, count in df.items():
        idf[word] = math.log((n_docs + 1) / (count + 1)) + 1
    return idf


_IDF = _build_idf()
_PRODS = {p.id: (p, d) for d in CATALOG for p in d.products}

# SENTIMENT

_POS_W = frozenset(stem(w) for w in (
    "вiдмiнно", "чудово", "гарно", "добре", "якiсно", "задоволений", "зручно",
    "надiйно", "рекомендую", "iдеально", "потужний",
    "excellent", "great", "good", "perfect", "recommend", "reliable", "convenient"
))

_NEG_W = frozenset(stem(w) for w in (
    "погано", "жахливо", "рваний", "крихкуватий", "шумнуватий", "дорого", "вiдкол", "проблема", "сiдає",
    "bad", "awful", "broken", "noisy", "expensive", "problem", "crack"
))


def sentiment(texts: list[str]) -> tuple[int, int, int, float]:
    pos = 0
    neg = 0
    neu = 0
    for text in texts:
        words_set = set(normalize(text))
        is_positive = bool(words_set & _POS_W)
        is_negative = bool(words_set & _NEG_W)

        if is_positive and not is_negative:
            pos += 1
        elif is_negative and not is_positive:
            neg += 1
        else:
            neu += 1

    score = round((pos - neg) / (len(texts) or 1), 2)
    return pos, neg, neu, score


# INTENT

_INTENTS = {
    "dept":    ["відділ", "вiддiл", "секці", "секцi", "department", "section", "departments"],
    "reviews": ["відгук", "вiдгук", "думк", "оцінк", "оцiнк", "review", "opinion", "feedback"],
    "compare": ["порівн", "порiвн", "різниц", "рiзниц", "compare", "versus", "vs"],
    "nlp":     ["токенізац", "токенiзац", "стемінг", "стемiнг", "nlp", "конвеєр", "pipeline", "pos"],
    "lang":    ["мова", "language", "мову"],
    "mode":    ["режим", "mode", "голос", "voice", "текст"],
    "help":    ["допомог", "help", "меню", "menu", "команд"],
    "exit":    ["вихід", "вихiд", "exit", "quit", "бувай", "goodbye", "bye", "стоп"],
}


def detect_intent(text: str) -> str:
    text_lower = text.lower()
    for intent_name, keywords in _INTENTS.items():
        if any(kw in text_lower for kw in keywords):
            return intent_name
    return "search"


def search(query: str, limit: int = 5) -> list[tuple[Product, Department, float]]:
    normalized_query = normalize(query)
    if not normalized_query:
        return []

    scored = []
    for d in CATALOG:
        for p in d.products:
            score = tfidf(normalized_query, normalize(p.text), _IDF)
            if score > 0:
                scored.append((p, d, score))

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:limit]


# VOICE

def speak(text: str):
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=LANG, slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tts.save(f.name)
            fname = f.name

        if sys.platform == "darwin":
            os.system(f"afplay '{fname}'")
        elif sys.platform == "win32":
            os.system(f"start /wait '' '{fname}'")
        else:
            os.system(f"mpg123 -q '{fname}' 2>/dev/null || ffplay -nodisp -autoexit '{fname}' 2>/dev/null")
        os.unlink(fname)
    except Exception as e:
        print(f"  [TTS error: {e}]")


def listen() -> str:
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        lang_code = "uk-UA" if LANG == "uk" else "en-US"

        print("  [listening...]")
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)

        text = recognizer.recognize_google(audio, language=lang_code)
        print(f"  [heard]: {text}")
        return text
    except Exception as e:
        print(f"  [STT error: {e}]")
        return ""


def output(text: str):
    print(text)
    if MODE == "voice":
        speak(text)


def get_input(prompt: str) -> str:
    if MODE == "voice":
        output(prompt)
        return listen()
    return input(prompt)


# UI

W = 60


def hr():
    print("-" * W)


def show_product(p: Product, d: Department, score: float | None = None):
    tag = f" [TF-IDF:{score:.3f}]" if score else ""
    output(f"\n  {p.name} [{p.id}]{tag}")
    output(f"  {d.name} | {d.zone}")
    output(f"  {p.price} / {p.unit} | stock: {p.stock}")
    output(f"  {p.desc}")


def show_reviews(code: str):
    entry = _PRODS.get(code.upper())
    if not entry:
        output(L("not_found"))
        return

    product, _ = entry
    output(L("reviews_hdr", product.name))
    hr()

    for i, review in enumerate(product.reviews, 1):
        output(f"  {i}. {review}")

    pos, neg, neu, score = sentiment(list(product.reviews))
    output(L("sentiment", pos, neu, neg, score))


def show_compare(code1: str, code2: str):
    entry1 = _PRODS.get(code1.upper())
    entry2 = _PRODS.get(code2.upper())

    if not entry1 or not entry2:
        output(L("not_found"))
        return

    p1, d1 = entry1
    p2, d2 = entry2

    vocab_set = set(normalize(p1.text)) | set(normalize(p2.text))
    vocab = sorted(vocab_set)

    def get_vector(product):
        tf = Counter(normalize(product.text))
        n = len(normalize(product.text)) or 1
        return [tf.get(w, 0) / n * _IDF.get(w, 0) for w in vocab]

    v1 = get_vector(p1)
    v2 = get_vector(p2)

    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude1 = math.sqrt(sum(a ** 2 for a in v1))
    magnitude2 = math.sqrt(sum(b ** 2 for b in v2))
    cosine_sim = dot_product / (magnitude1 * magnitude2 + 1e-9)

    _, _, _, s1 = sentiment(list(p1.reviews))
    _, _, _, s2 = sentiment(list(p2.reviews))

    output(L("compare_hdr", p1.name, p2.name))
    hr()

    comparisons = [
        ("Code", p1.id, p2.id),
        ("Price", f"{p1.price}", f"{p2.price}"),
        ("Stock", f"{p1.stock} {p1.unit}", f"{p2.stock} {p2.unit}")
    ]

    for label, val1, val2 in comparisons:
        output(f"  {label:<10} {str(val1):<22} {val2}")

    output(L("cos_sim", cosine_sim))

    cheaper_name = p1.name if p1.price < p2.price else p2.name
    output(L("cheaper", cheaper_name))

    better_reviews_name = "="
    if s1 > s2:
        better_reviews_name = p1.name
    elif s2 > s1:
        better_reviews_name = p2.name
    output(L("better_rev", better_reviews_name))


def show_nlp(text: str):
    tokens = tokenize(text)
    filtered = [t for t in tokens if t not in (STOPWORDS["uk"] | STOPWORDS["en"])]
    tagged = pos_tag(tokens)
    stems = [stem(t) for t in filtered]

    tf_map = Counter(normalize(text))
    n = len(normalize(text)) or 1

    vocab_top = sorted(_IDF, key=lambda w: -_IDF[w])[:5]
    vector = {w: round(tf_map.get(w, 0) / n * _IDF.get(w, 0), 4) for w in vocab_top}

    pos, neg, neu, score = sentiment([text])
    top_results = search(text, limit=1)

    verdict = "neutral"
    if score > 0:
        verdict = "positive"
    elif score < 0:
        verdict = "negative"

    output(L("nlp_hdr", text))
    hr()
    output(f"  1. tokenize  -> {tokens}")
    output(f"  2. stopwords -> {filtered}")
    output(f"  3. POS (pymorphy3):")

    for word, pos_lbl, lemma in tagged:
        output(f"     {word:<16} {pos_lbl:<6} lemma:{lemma}")

    output(f"  4. stem      -> {stems}")
    output(f"  5. TF top-3  -> {dict(Counter(stems).most_common(3))}")
    output(f"  6. TF-IDF vec-> {vector}")
    output(f"  7. sentiment -> {verdict} (score={score:+.2f})")
    output(f"  8. intent    -> {detect_intent(text)}")

    if top_results:
        top_product = top_results[0][0]
        top_score = top_results[0][2]
        output(f"  9. TF-IDF#1  -> {top_product.name} ({top_score:.3f})")


# REPL

_RE_DEPT    = re.compile(r"(?:відділ|вiддiл|department)\s+([1-4])$", re.I)
_RE_REVIEWS = re.compile(r"(?:відгуки?|вiдгуки?|reviews?)\s+([a-dA-D]\d{2})\b", re.I)
_RE_CODE    = re.compile(r"(?:товар|product)\s+([a-dA-D]\d{2})\b", re.I)
_RE_COMPARE = re.compile(r"(?:порівняти|порiвняти|compare)\s+([a-dA-D]\d{2})\s+([a-dA-D]\d{2})\b", re.I)
_RE_NLP     = re.compile(r"nlp\s+(.+)", re.I)
_RE_ITEM    = re.compile(r"(?:товар|product)\s+(.+)", re.I)


def dispatch(raw: str) -> bool:
    global LANG, MODE
    cmd = raw.strip()
    if not cmd:
        return True

    low = cmd.lower()
    intent = detect_intent(cmd)

    if intent == "exit":
        output(L("bye"))
        return False

    if intent == "help":
        output(L("help"))
        return True

    if intent == "lang":
        LANG = "en" if LANG == "uk" else "uk"
        output(L("lang_switch"))
        return True

    if intent == "mode":
        MODE = "voice" if MODE == "text" else "text"
        output(L("mode_switch", MODE))
        return True

    if intent == "dept" and not _RE_DEPT.match(low):
        for d in CATALOG:
            output(L("dept_hdr", d.id, d.name, d.zone))
            output(f"  {d.desc}")
        return True

    match_compare = _RE_COMPARE.match(cmd)
    if match_compare:
        show_compare(match_compare.group(1), match_compare.group(2))
        return True

    match_dept = _RE_DEPT.match(low)
    if match_dept:
        dept_id = match_dept.group(1)
        dept = None
        for d in CATALOG:
            if d.id == dept_id:
                dept = d
                break

        if dept:
            output(f"\n{dept.name} | {dept.zone}")
            hr()
            for p in dept.products:
                output(f"  {p.id:<5} {p.name:<30} {p.price} / {p.unit}")
        return True

    match_reviews = _RE_REVIEWS.match(cmd)
    if match_reviews:
        show_reviews(match_reviews.group(1))
        return True

    match_code = _RE_CODE.match(cmd)
    if match_code:
        entry = _PRODS.get(match_code.group(1).upper())
        if entry:
            show_product(entry[0], entry[1])
        else:
            output(L("not_found"))
        return True

    match_nlp = _RE_NLP.match(cmd)
    if match_nlp:
        show_nlp(match_nlp.group(1))
        return True

    match_item = _RE_ITEM.match(cmd)
    if match_item:
        results = search(match_item.group(1))
        if results:
            output(L("found", len(results)))
            for p, d, s in results:
                show_product(p, d, s)
        else:
            output(L("no_results", match_item.group(1)))
        return True

    # Default fallback to general search
    results = search(cmd)
    if results:
        output(L("found", len(results)))
        for p, d, s in results:
            show_product(p, d, s)
    else:
        output(L("no_results", cmd))

    return True


# MAIN

def main():
    global LANG, MODE
    print("=" * W)
    print("  BudMarket Plus | Buyer Guide")
    print("  NLP: tokenize / POS / stem / TF-IDF / sentiment / intent")
    print("=" * W)

    lang_in = input("  Language / Мова [uk/en, default=uk]: ").strip().lower()
    if lang_in in ("en", "english"):
        LANG = "en"

    mode_in = input("  Mode / Режим [text/voice, default=text]: ").strip().lower()
    if mode_in in ("voice", "голос"):
        MODE = "voice"

    print(f"  >> {LANG.upper()} | {MODE.upper()}\n")

    name = get_input(L("ask_name"))
    output(L("greet", name or "..."))
    output(L("help"))

    while True:
        try:
            user_input = get_input(L("prompt"))
            if not dispatch(user_input):
                break
        except (KeyboardInterrupt, EOFError):
            output(L("bye"))
            break


if __name__ == "__main__":
    main()