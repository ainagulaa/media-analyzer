import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv("data/articles_cleaned.csv", encoding="utf-8")

if "cleaned_text" not in df.columns:
    raise ValueError("❌ В файле должна быть колонка 'cleaned_text'")

texts = df["cleaned_text"].fillna("").astype(str).tolist()


vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts)

kmeans_topics = KMeans(n_clusters=5, random_state=42)
df["topic_cluster"] = kmeans_topics.fit_predict(X)

kmeans_emotion = KMeans(n_clusters=3, random_state=42)
df["emotion_cluster"] = kmeans_emotion.fit_predict(X)

output_path = "data/articles_clustered.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"✅ Кластеризация завершена. Результат: {output_path}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Кластеры тематики")
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["topic_cluster"], cmap="tab10")
plt.subplot(1, 2, 2)
plt.title("Кластеры тональности")
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["emotion_cluster"], cmap="Set2")
plt.tight_layout()
plt.show()
