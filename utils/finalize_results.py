# -*- coding: utf-8 -*-
"""
ФАЙЛ: finalize_results.py

Назначение:
    - Добавляет оценку влияния статьи (низкое, среднее, высокое)
    - Объединяет все данные в итоговый CSV
"""

import pandas as pd

# ————————————————————————————————————————
# 📥 Загрузка кластеризованных данных
# ————————————————————————————————————————

df = pd.read_csv("data/articles_clustered.csv", encoding="utf-8")

# ————————————————————————————————————————
# 🔍 Оценка степени влияния
# ————————————————————————————————————————

def estimate_influence(text):
    """
    Простая эвристика:
    - Высокое влияние: длина текста > 300 или есть ключевые слова
    - Среднее: 150–300
    - Низкое: < 150
    """
    text = str(text)
    length = len(text)

    high_keywords = ["президент", "министр", "протест", "кризис", "скандал", "реформа"]
    count_keywords = sum(1 for word in high_keywords if word in text.lower())

    if length > 300 or count_keywords >= 2:
        return "высокое"
    elif length > 150:
        return "среднее"
    else:
        return "низкое"

df["influence_score"] = df["text"].apply(estimate_influence)

# ————————————————————————————————————————
# 💾 Сохраняем результат
# ————————————————————————————————————————

output_path = "data/articles_final.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"✅ Итоговая таблица сохранена: {output_path}")
