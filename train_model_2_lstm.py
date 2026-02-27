import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DATASET_FILE = "dataset_final.jsonl"
MAX_WORDS = 20000
MAX_LEN = 150
EMBEDDING_DIM = 64

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

    print("Tokenizer hazırlanıyor...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<UNK>")
    tokenizer.fit_on_texts(X_train)

    print("Metinler sayısallaştırılıyor...")
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    print("Padding uygulanıyor...")
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN)

    print("LSTM modeli oluşturuluyor...")
    model = Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    print("Model eğitiliyor...")
    es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    history = model.fit(
        X_train_pad, y_train,
        validation_split=0.2,
        epochs=5,
        batch_size=64,
        callbacks=[es]
    )

    print("Tahmin yapılıyor...")
    y_pred = (model.predict(X_test_pad) > 0.5).astype("int32")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("======================================")
    print("MODEL 2 SONUÇLARI (LSTM)")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)
    print("\nConfusion Matrix:")
    print(cm)
    print("======================================")

    print("Model kaydediliyor...")
    model.save("model_2_lstm.h5")

    print("Tokenizer kaydediliyor...")
    import joblib
    joblib.dump(tokenizer, "tokenizer_lstm.pkl")

    print("model_2_lstm.h5 ve tokenizer_lstm.pkl oluşturuldu!")

if __name__ == "__main__":
    main()
