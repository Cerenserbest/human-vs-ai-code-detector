# Human vs. AI Code Detector 🤖💻

Bu proje, bir kod parçacığının bir insan tarafından mı yazıldığını yoksa yapay zeka (LLM) tarafından mı oluşturulduğunu tespit etmek için geliştirilmiştir. Proje kapsamında farklı makine öğrenmesi ve derin öğrenme mimarileri karşılaştırmalı olarak kullanılmıştır.

## 🚀 Özellikler
* **Çoklu Model Desteği:** TF-IDF + Logistic Regression, LSTM ve GRU modelleri.
* **Veri Seti:** GitHub üzerinden çekilen gerçek insan kodları ve yapay zeka tarafından üretilen sentetik kodlar.
* **Web Arayüzü:** Flask tabanlı basit ve kullanışlı bir kullanıcı arayüzü.

## 🛠️ Kullanılan Teknolojiler
* **Dil:** Python
* **Kütüphaneler:** TensorFlow, Keras, Scikit-learn, Pandas, Flask
* **Modeller:** * LSTM (Long Short-Term Memory)
  * GRU (Gated Recurrent Unit)
  * TF-IDF Vectorizer

## 📂 Proje Yapısı
- `app.py`: Web arayüzünü çalıştıran Flask uygulaması.
- `train_model_2_lstm.py`: LSTM modelinin eğitim betiği.
- `model_2_lstm.h5`: Eğitilmiş LSTM model dosyası.
- `tokenizer_lstm.pkl`: Metin verilerini sayısal verilere dönüştüren sözlük yapısı.
- `dataset_final.jsonl`: Eğitim ve test için kullanılan veri seti.

## 📊 Kurulum ve Çalıştırma

1. Projeyi bilgisayarınıza indirin:
   ```bash
   git clone [https://github.com/Cerenserbest/human-vs-ai-code-detector.git](https://github.com/Cerenserbest/human-vs-ai-code-detector.git)
   cd human-vs-ai-code-detector
