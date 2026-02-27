import json
import random
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DATASET_FILE = "dataset_final.jsonl"

def load_jsonl(path):
    texts = []
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            # Artık label zaten 0 veya 1 (int)
            labels.append(int(obj["label"]))
    return texts, np.array(labels)


def main():
    print("Dataset yükleniyor...")
    texts, labels = load_jsonl(DATASET_FILE)
    print(f"Toplam örnek: {len(texts)}")

    print("Train/Test split yapılıyor...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print("TF-IDF vectorizer oluşturuluyor...")
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        lowercase=True,
        stop_words=None
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Model eğitiliyor (Logistic Regression)...")
    model = LogisticRegression(max_iter=2000, class_weight="balanced")

    model.fit(X_train_vec, y_train)

    print("Tahmin yapılıyor...")
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("======================================")
    print("MODEL 1 SONUÇLARI (TF-IDF + Logistic)")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)
    print("\nConfusion Matrix:")
    print(cm)
    print("======================================")

    print("Model kaydediliyor...")
    joblib.dump(model, "model_1.pkl")
    joblib.dump(vectorizer, "vectorizer_1.pkl")
    print("model_1.pkl ve vectorizer_1.pkl oluşturuldu!")

if __name__ == "__main__":
    main()
