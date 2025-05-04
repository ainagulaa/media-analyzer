# -*- coding: utf-8 -*-
"""
ФАЙЛ: text_preprocessing.py

Назначение:
    - Загружает CSV-файл с колонкой 'text'
    - Очищает тексты (русский + казахский)
    - Сохраняет результат в новый CSV с колонкой 'cleaned_text'

Команды:
    pip install pandas nltk
"""

import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ————————————————————————————————————————
# 🔧 Загрузка ресурсов NLTK
# ————————————————————————————————————————
def download_nltk_resources():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

download_nltk_resources()

# ————————————————————————————————————————
# 📌 Стоп-слова
# ————————————————————————————————————————

russian_stopwords = set(stopwords.words("russian"))

kazakh_stopwords = {
    "мен", "және", "бір", "осы", "сол", "бұл", "бар", "екен", "етіп", "қатты", "еді", "болды",
    "болып", "туралы", "сияқты", "тағы", "көп", "менің", "сен", "ол", "тағыда", "едің", "едіңіз",
    "соң", "сонымен", "деген", "арасында", "береді", "қалды", "дегенмен", "керек", "қажет", "қандай"
}

# ————————————————————————————————————————
# ✨ Функция очистки текста
# ————————————————————————————————————————
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)      # удаление пунктуации
    text = re.sub(r"\d+", "", text)          # удаление цифр

    try:
        words = word_tokenize(text, language='russian')  # подходит и для казахского
    except:
        words = text.split()

    filtered_words = [
        word for word in words
        if word not in russian_stopwords and word not in kazakh_stopwords
    ]

    return " ".join(filtered_words)

# ————————————————————————————————————————
# 📥 Загрузка и обработка CSV
# ————————————————————————————————————————
def process_csv(input_path: str, output_path: str):
    try:
        df = pd.read_csv(input_path, encoding="utf-8")
    except FileNotFoundError:
        print(f"❌ Файл не найден: {input_path}")
        return

    if "text" not in df.columns:
        print("❌ Входной файл должен содержать колонку 'text'")
        return

    print("🔄 Очистка текстов...")

    df["cleaned_text"] = df["text"].astype(str).apply(preprocess_text)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ Готово. Сохранено в: {output_path}")

# ————————————————————————————————————————
# 🚀 Запуск скрипта
# ————————————————————————————————————————
if __name__ == "__main__":
    input_csv = "data/articles.csv"
    output_csv = "data/articles_cleaned.csv"
    process_csv(input_csv, output_csv)
