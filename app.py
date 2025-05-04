import streamlit as st
st.set_page_config(page_title="Медиа-Анализатор", layout="wide")

import pandas as pd
import re
import nltk
import plotly.express as px
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from bs4 import BeautifulSoup
import requests
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import defaultdict
import torch
import math

# —————————————————————————————
nltk.download("punkt")
nltk.download("stopwords")
stemmer = SnowballStemmer("russian")
ru_stop = set(stopwords.words("russian"))
kz_stop = {
    "мен", "және", "бір", "осы", "сол", "бұл", "бар", "екен", "етіп", "қатты", "еді", "болды",
    "болып", "туралы", "сияқты", "тағы", "көп", "менің", "сен", "ол", "тағыда", "едің", "едіңіз"
}

@st.cache_resource
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny-sentiment-balanced")
    model = AutoModelForSequenceClassification.from_pretrained("cointegrated/rubert-tiny-sentiment-balanced")
    return tokenizer, model

sent_tokenizer, sent_model = load_sentiment_model()


@st.cache_resource
def load_fake_news_model():
    tokenizer = AutoTokenizer.from_pretrained("tellowit/rubert-fake-news-classification")
    model = AutoModelForSequenceClassification.from_pretrained("tellowit/rubert-fake-news-classification")
    return tokenizer, model

fake_tokenizer, fake_model = load_fake_news_model()

def detect_fake_rubert(text):
    inputs = fake_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    with torch.no_grad():
        outputs = fake_model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    fake_conf = probs[0][1].item()  # вероятность, что это фейк

    if fake_conf > 0.85:
        return f"🚨 Фейк (уверенность: {fake_conf:.0%})"
    elif fake_conf > 0.6:
        return f"⚠️ Возможно фейк (уверенность: {fake_conf:.0%})"
    elif fake_conf > 0.4:
        return f"❓ Сомнительно (уверенность: {fake_conf:.0%})"
    else:
        return f"✅ Надёжный (уверенность: {1 - fake_conf:.0%})"

def clean(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    return " ".join([w for w in tokens if w not in ru_stop and w not in kz_stop])

def estimate_influence(text, views=None, likes=None, shares=None):
    text = text.lower()
    words = text.split()
    word_count = len(words)

    if word_count == 0:
        return "🟢 Низкое"

    # Стемминг текста
    stemmed_text = [stemmer.stem(word) for word in words]

    # Ключевые категории
    high_risk = ["взрыв", "террор", "митинг", "санкци", "военн", "стрельб", "авар", "угроз", "паник", "теракт", "убийств", "катастроф"]
    medium_risk = ["реформ", "закон", "инфляц", "кризис", "штраф", "суд", "петици", "жалоб", "арест", "расследован", "обострен", "дефицит"]
    emotional = ["гнев", "ярост", "шок", "восторг", "возмущен", "скандал", "треб", "обвин", "негодован", "беспокойств", "жестк", "резк", "позор"]
    clickbait = ["шок", "неожид", "взорвал", "невероятн", "эксклюзив", "срочн", "шокирующ"]

    # Подсчёты
    def count_matches(stemmed_list, category):
        return sum(1 for word in stemmed_list if any(word.startswith(trigger) for trigger in category))

    score = 0
    score += count_matches(stemmed_text, high_risk) * 3
    score += count_matches(stemmed_text, medium_risk) * 2
    score += count_matches(stemmed_text, emotional) * 1.5
    score += count_matches(stemmed_text, clickbait) * 1.2

    # Плотность ключевых слов
    total_triggers = count_matches(stemmed_text, high_risk + medium_risk + emotional)
    density = total_triggers / word_count
    score += density * 10  # масштабируем

    # Дополнительные сигналы
    score += 1 if text.count("!") > 2 else 0
    score += 0.5 if text.count("?") > 2 else 0
    score += math.log(word_count + 1, 10)

    # Социальные метрики
    if views and views > 10000:
        score += 1.5
    if likes and likes > 500:
        score += 1
    if shares and shares > 100:
        score += 1

    # Финальный вывод
    if score >= 12:
        return "🚨 Очень высокое"
    elif score >= 8:
        return "🔺 Высокое"
    elif score >= 4:
        return "⚠️ Среднее"
    else:
        return "🟢 Низкое"

def classify_topic(text):
    text = text.lower()

    topic_keywords = {
        "⚖️ Политика": {
            "президент": 3, "депутат": 3, "парламент": 3, "закон": 3, "выбор": 2,
            "голосовани": 2, "правительств": 3, "партий": 2, "митинг": 3, "оппозици": 2
        },
        "💰 Экономика": {
            "экономик": 3, "инфляц": 3, "курс": 2, "бюджет": 2, "деньг": 1,
            "финанс": 3, "налог": 2, "банк": 2, "зарплат": 2, "цен": 1
        },
        "🔬 Наука/Технологии": {
            "наук": 3, "инновац": 3, "технолог": 3, "исследован": 3, "алгоритм": 2,
            "лаборатор": 2, "робот": 2, "ИИ": 3, "искусственн": 3, "нейросет": 2
        },
        "🌍 Экология": {
            "эколог": 3, "природ": 2, "климат": 3, "углерод": 2, "загрязн": 3,
            "окружающ": 2, "свалк": 2, "водоём": 1, "глобальн потеплен": 3
        },
        "🎓 Образование": {
            "образовани": 3, "школ": 3, "учеб": 2, "студент": 2, "университет": 2,
            "экзамен": 1, "учител": 1, "реформа": 1, "академ": 1
        },
        "🧑‍⚕️ Здравоохранение": {
            "медицин": 3, "вакцин": 3, "врач": 2, "лечен": 2, "эпидеми": 2,
            "здравоохранен": 3, "пациент": 1, "вирус": 2, "болезн": 2, "симптом": 1
        },
        "🕊️ Общество": {
            "социальн": 2, "семь": 2, "пенсионер": 2, "молодеж": 2, "инклюз": 1,
            "равенств": 2, "общественн": 2, "населен": 1, "граждан": 1
        },
        "🎭 Культура/Искусство": {
            "культур": 3, "искусств": 3, "музе": 2, "театр": 2, "фестиваль": 1,
            "традиц": 2, "наследи": 2, "литератур": 1, "премь": 1, "кино": 2
        },
        "📰 СМИ/Медиа": {
            "журналист": 3, "репорт": 2, "цензур": 2, "сми": 3, "интервью": 1,
            "новостн": 2, "редакц": 2, "соцсет": 2, "медиа": 2, "канал": 1
        },
        "⚽ Спорт": {
            "спорт": 3, "футбол": 3, "матч": 2, "команд": 2, "игрок": 2,
            "чемпионат": 2, "турнир": 2, "олимпиад": 2, "тренер": 1
        },
        "🚨 Происшествия": {
            "пожар": 3, "авар": 3, "дтп": 3, "убийств": 3, "катастроф": 2,
            "чп": 2, "наезд": 1, "задержан": 2, "преступлен": 2, "инцидент": 1
        },
        "🌐 Международные дела": {
            "международн": 3, "санкц": 3, "конфликт": 3, "вторжен": 3, "войн": 3,
            "нато": 2, "сша": 2, "украин": 3, "евросоюз": 2, "путин": 2, "байден": 2
        }
    }

    scores = defaultdict(int)

    for topic, keywords in topic_keywords.items():
        for root, weight in keywords.items():
            if re.search(rf"\b{root}\w*", text):  # ищет по корням
                scores[topic] += weight

    if not scores:
        return "📦 Прочее"

    max_score = max(scores.values())
    candidates = [t for t, s in scores.items() if s == max_score]

    if len(candidates) == 1:
        return candidates[0]

    # Разрешаем равенство — выбираем по количеству ключей
    key_matches = {
        topic: sum(1 for word in topic_keywords[topic] if re.search(rf"\b{word}\w*", text))
        for topic in candidates
    }

    return max(key_matches, key=key_matches.get)


def analyze_sentiment(text):
    positive = ["достижение", "успех", "развитие", "рост", "поддержка"]
    negative = ["кризис", "протест", "проблема", "инцидент", "угроза"]
    pos_count = sum(1 for word in positive if word in text)
    neg_count = sum(1 for word in negative if word in text)
    if pos_count > neg_count:
        return "🙂 Позитив"
    elif neg_count > pos_count:
        return "☹️ Негатив"
    else:
        return "😐 Нейтрально"

def analyze_sentiment_model(text):
    inputs = sent_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    with torch.no_grad():
        outputs = sent_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    label = torch.argmax(probs).item()

    if label == 0:
        return "☹️ Негатив"
    elif label == 1:
        return "😐 Нейтрально"
    else:
        return "🙂 Позитив"
    
def hybrid_sentiment(tone_heuristic, tone_model):
    if tone_heuristic == tone_model:
        return tone_model  # Совпадают — берём любую
    elif tone_model != "😐 Нейтрально":
        return tone_model  # Если модель уверена — доверяем ей
    else:
        return tone_heuristic  # Иначе fallback на эвристику



def fetch_article(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/90 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs)
        return text if len(text) > 100 else "Не удалось извлечь содержимое."
    except:
        return "Не удалось получить статью."

def visualize_clusters(X, labels, title, label_map=None):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X.toarray())
    
    # если label_map передан — применяем его, иначе оставляем как есть
    if label_map:
        named_labels = pd.Series(labels).map(label_map)
    else:
        named_labels = pd.Series(labels)

    df_vis = pd.DataFrame({
        "PCA1": reduced[:, 0],
        "PCA2": reduced[:, 1],
        "Категория": named_labels
    })

    fig = px.scatter(
        df_vis, x="PCA1", y="PCA2", color="Категория",
        title=title, template="plotly_white", height=500,
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    st.plotly_chart(fig, use_container_width=True)

# ——— Интерфейс ———
st.title("🧠 Интеллектуальная система определения степени воздействия медиа-контента на общественное мнение")
mode = st.radio("Выберите режим:", ["🔎 Вставить URL статьи", "📁 Загрузить CSV-файл"])

if mode == "🔎 Вставить URL статьи":
    url = st.text_input("Вставьте ссылку на статью")
    if url:
        article_text = fetch_article(url)
        st.text_area("Текст статьи", article_text[:5000], height=300)

        cleaned = clean(article_text)
        topic = classify_topic(cleaned)
        tone = analyze_sentiment(cleaned)
        model_tone = analyze_sentiment_model(cleaned)
        st.markdown(f"**🤖 Модельная тональность:** {model_tone}")
        if model_tone != tone:
            st.markdown("⚠️ Тональности **расходятся**")
        else:
            st.markdown("✅ Тональности **совпадают**")
        hybrid_tone = hybrid_sentiment(tone, model_tone)
        st.markdown(f"**🧪 Гибридная тональность:** {hybrid_tone}")

        infl = estimate_influence(article_text)
        fake_flag = detect_fake_rubert(article_text)

        st.markdown(f"**🧭 Тематика:** {topic}")
        st.markdown(f"**😐 Тональность:** {tone}")
        st.markdown(f"**📣 Влияние:** {infl}")
        st.markdown(f"**🚩 Фейковость:** {fake_flag}")

elif mode == "📁 Загрузить CSV-файл":
    uploaded_file = st.file_uploader("Загрузите CSV с колонкой 'text'", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "text" not in df.columns:
            st.error("CSV должен содержать колонку 'text'")
        else:
            with st.spinner("Анализируем..."):
                df["cleaned_text"] = df["text"].apply(clean)
                tfidf = TfidfVectorizer(max_features=1000)
                X = tfidf.fit_transform(df["cleaned_text"])

                # Кластеризация по тематикам (12 кластеров)
                topic_model = KMeans(n_clusters=12, random_state=42)
                df["topic_cluster"] = topic_model.fit_predict(X)
                df["topic_label"] = "🧩 Кластер " + df["topic_cluster"].astype(str)

                # Кластеризация по тональности
                tone_model = KMeans(n_clusters=3, random_state=2)
                df["tone_cluster"] = tone_model.fit_predict(X)
                tone_map = {
                    0: "🙂 Позитив",
                    1: "😐 Нейтрально",
                    2: "☹️ Негатив"
                }
                df["tone_label"] = df["tone_cluster"].map(tone_map)

                # Остальные эвристики
                df["influence"] = df["text"].apply(estimate_influence)
                df["fake_flag"] = df["text"].apply(detect_fake_rubert)

                tone_labels = {
                    0: "🙂 Позитив",
                    1: "😐 Нейтрально",
                    2: "☹️ Негатив"
                }

                df["topic_label"] = df["cleaned_text"].apply(classify_topic)
                df["tone_label"] = df["tone_cluster"].map(tone_labels)
                df["tone_model"] = df["cleaned_text"].apply(analyze_sentiment_model)
                df["hybrid_tone"] = df.apply(lambda row: hybrid_sentiment(row["tone_label"], row["tone_model"]), axis=1)
                df["tone_match"] = df["tone_model"] == df["tone_label"]


                sil_score = silhouette_score(X, df["topic_cluster"])
                calinski = calinski_harabasz_score(X.toarray(), df["topic_cluster"])
                davies = davies_bouldin_score(X.toarray(), df["topic_cluster"])

                metrics_df = pd.DataFrame({
                    "Метрика": ["Silhouette Score", "Calinski-Harabasz Index", "Davies-Bouldin Index"],
                    "Значение": [round(sil_score, 4), round(calinski, 2), round(davies, 4)]
                })

                st.markdown("### 📐 Метрики качества кластеризации")
                st.dataframe(metrics_df, use_container_width=True)

                # Интерпретация
                explanations = []

                if sil_score >= 0.5:
                    explanations.append(f"• **Silhouette Score** = {sil_score:.4f}: отличное разделение кластеров.")
                elif sil_score >= 0.3:
                    explanations.append(f"• **Silhouette Score** = {sil_score:.4f}: приемлемое качество кластеризации.")
                else:
                    explanations.append(f"• **Silhouette Score** = {sil_score:.4f}: слабое разделение кластеров, возможно, стоит пересмотреть количество кластеров.")

                if calinski >= 100:
                    explanations.append(f"• **Calinski-Harabasz Index** = {calinski:.2f}: высокое межкластерное расстояние, хорошая структура.")
                else:
                    explanations.append(f"• **Calinski-Harabasz Index** = {calinski:.2f}: умеренная кластерная структура.")

                if davies <= 1:
                    explanations.append(f"• **Davies-Bouldin Index** = {davies:.3f}: кластеры хорошо разделены и плотные.")
                else:
                    explanations.append(f"• **Davies-Bouldin Index** = {davies:.3f}: возможны пересечения или размытые границы между кластерами.")

                st.markdown("### 🧠 Интерпретация метрик:")
                for line in explanations:
                    st.markdown(line)


            st.success("✅ Анализ завершён!")
            st.subheader("📣 Распределение влияния медиа-контента")
            influence_counts = df["influence"].value_counts().reset_index()
            influence_counts.columns = ["Влияние", "Количество"]

            fig_inf = px.bar(
                influence_counts,
                x="Влияние",
                y="Количество",
                color="Влияние",
                color_discrete_map={
                    "высокое": "crimson",
                    "среднее": "orange",
                    "низкое": "green"
                },
                text="Количество",
                title="Распределение по степени влияния",
            )
            fig_inf.update_layout(xaxis_title=None, yaxis_title="Кол-во статей", title_x=0.5)
            st.plotly_chart(fig_inf, use_container_width=True)
            match_rate = df["tone_match"].mean()
            st.markdown(f"### 🔍 Совпадение эвристики и модели тональности: **{match_rate:.1%}**")


            st.dataframe(df[["text", "topic_label", "tone_label", "tone_model", "hybrid_tone", "tone_match", "influence", "fake_flag"]], use_container_width=True)
            csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button("📥 Скачать результат", csv_bytes, "articles_final.csv", "text/csv")

            st.markdown("### 📊 Визуализация кластеров")
            st.markdown("#### Тематика")
            visualize_clusters(X, df["topic_label"], "Тематика", None)
            st.markdown("#### Тональность")
            visualize_clusters(X, df["tone_cluster"], "Тональность", tone_labels)