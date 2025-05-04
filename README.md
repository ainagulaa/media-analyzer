# 🧠 Интеллектуальная система анализа медиа-контента

Этот проект представляет собой Streamlit-приложение, способное автоматически анализировать текстовые статьи (из URL или CSV), определяя:

- **Тематику публикации**
- **Тональность (по эвристике и модели RuSentiment)**
- **Степень воздействия на общественное мнение**
- **Вероятность фейковости (модель RuBERT)**
- **Кластеризацию по темам и тону**
- **Метрики качества кластеризации**
- **Сравнение результатов нескольких методов оценки**

---

## 🚀 Возможности

🔍 Анализ одной статьи по ссылке  
📁 Массовый анализ загруженного CSV-файла  
📣 Оценка влияния статьи (эвристика с учётом ключевых слов и соц.сигналов)  
🧭 Тематическая классификация  
😐 Оценка тональности: эвристическая + модельная  
🚩 Обнаружение фейковых новостей  
📊 Визуализация кластеров  
📥 Экспорт результатов в CSV  

---

## 🛠️ Используемые технологии

- `Python 3.10+`
- `Streamlit`
- `scikit-learn`
- `transformers (HuggingFace)`
- `nltk`, `re`, `math`
- `plotly`, `pandas`, `BeautifulSoup`

Модели:
- [RuSentiment (`cointegrated/rubert-tiny-sentiment-balanced`)](https://huggingface.co/cointegrated/rubert-tiny-sentiment-balanced)
- [Fake News Detector (`tellowit/rubert-fake-news-classification`)](https://huggingface.co/tellowit/rubert-fake-news-classification)

---

## 📦 Установка

```bash
git clone https://github.com/your-username/media-impact-analyzer.git
cd media-impact-analyzer
pip install -r requirements.txt
streamlit run app.py