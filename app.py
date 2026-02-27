from flask import Flask, request, jsonify, render_template

import joblib
import numpy as np
import tensorflow as tf

# ------------------------------
# MODEL 1 (TF-IDF + Logistic Regression)
# ------------------------------
model_1 = joblib.load("model_1.pkl")
vectorizer_1 = joblib.load("vectorizer_1.pkl")

# ------------------------------
# MODEL 2 (LSTM)
# ------------------------------
model_2 = tf.keras.models.load_model("model_2_lstm.h5")
tokenizer_lstm = joblib.load("tokenizer_lstm.pkl")

# ------------------------------
# MODEL 3 (GRU)
# ------------------------------
model_3 = tf.keras.models.load_model("model_3_gru.h5")
tokenizer_gru = joblib.load("tokenizer_gru.pkl")

MAX_LEN = 150

app = Flask(__name__)


def predict_lr_proba(text):
    vector = vectorizer_1.transform([text])
    proba = model_1.predict_proba(vector)[0][1]   # 1 = AI ihtimali
    return float(proba)


def predict_lstm_proba(text):
    seq = tokenizer_lstm.texts_to_sequences([text])
    pad = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN)
    proba = float(model_2.predict(pad)[0][0])     # 0-1 arası probability
    return proba


def predict_gru_proba(text):
    seq = tokenizer_gru.texts_to_sequences([text])
    pad = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN)
    proba = float(model_3.predict(pad)[0][0])     # 0-1 arası probability
    return proba


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    code = data.get("code")

    if not code:
        return jsonify({"error": "No code provided"}), 400

    # Her model AI olma olasılığı (0-1 arası)
    p1 = predict_lr_proba(code)
    p2 = predict_lstm_proba(code)
    p3 = predict_gru_proba(code)

    # Final probability = 3 modelin ortalaması
    final_p = (p1 + p2 + p3) / 3
    final_label = "AI" if final_p >= 0.5 else "Human"

    response = {
        "input": code,
        "model_1_lr_proba": p1,
        "model_2_lstm_proba": p2,
        "model_3_gru_proba": p3,
        "final_probability": final_p,
        "final_label": final_label
    }

    return jsonify(response)

@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
