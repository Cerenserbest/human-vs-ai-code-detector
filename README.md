# 🤖 Human vs AI Code Detector

## 🧠 Proje Hakkında
Bu proje, bir kod parçacığının bir insan tarafından mı yazıldığını yoksa yapay zeka (LLM) tarafından mı oluşturulduğunu tespit etmek için geliştirilmiş bir **Doğal Dil İşleme (NLP)** sınıflandırma sistemidir. Proje kapsamında farklı makine öğrenmesi ve derin öğrenme mimarileri (LSTM, GRU, TF-IDF) karşılaştırmalı olarak kullanılmıştır.

## 🚀 Özellikler
* **Gelişmiş Veri Önişleme:** GitHub üzerinden çekilen gerçek insan kodları ve yapay zeka tarafından üretilen kodlar üzerinde temizleme ve birleştirme süreçleri.
* **Çoklu Model Mimarisi:**
    * **TF-IDF + Classical ML:** Hızlı ve etkili geleneksel yaklaşım.
    * **LSTM (Long Short-Term Memory):** Kodun ardışık yapısını ve mantıksal akışını anlayan derin öğrenme modeli.
    * **GRU (Gated Recurrent Unit):** LSTM'e alternatif, verimli ve modern RNN mimarisi.
* **Kalıcı Model Saklama:** Tokenizer ve vektörizer dosyalarının `.pkl` formatında saklanarak web arayüzüne entegrasyonu.
* **Kullanıcı Arayüzü:** Flask tabanlı, herkesin kullanımına uygun web arayüzü (`app.py`).

## 🛠️ Kullanılan Teknolojiler
* **Dil:** Python
* **Derin Öğrenme:** TensorFlow / Keras
* **Makine Öğrenmesi:** Scikit-learn
* **Web Framework:** Flask
* **Veri İşleme:** Pandas, JSONL, Pickle

## 📂 Proje Yapısı
```text
humanai/
│
├── dataset_final.jsonl        # Birleştirilmiş ana eğitim veri seti
├── human_code_snippets.jsonl  # Kaynak insan kodları veri seti (Yerel)
│
├── model_1.pkl                # TF-IDF tabanlı model dosyası
├── model_2_lstm.h5            # Eğitilmiş LSTM model dosyası
├── model_3_gru.h5             # Eğitilmiş GRU model dosyası
│
├── tokenizer_lstm.pkl         # LSTM için kelime sözlüğü
├── tokenizer_gru.pkl          # GRU için kelime sözlüğü
├── vectorizer_1.pkl           # TF-IDF vektörizer dosyası
│
├── app.py                     # Web arayüzünü başlatan Flask uygulaması
├── train_model_2_lstm.py      # LSTM modeli eğitim betiği
├── train_model_3_gru.py       # GRU modeli eğitim betiği
└── veri_cekme_human.py        # GitHub API üzerinden veri toplama kodu
▶️ Nasıl Çalıştırılır?
1. Projeyi Klonlayın
Bash
git clone [https://github.com/Cerenserbest/human-vs-ai-code-detector.git](https://github.com/Cerenserbest/human-vs-ai-code-detector.git)
cd human-vs-ai-code-detector
2. Gerekli Kütüphaneleri Yükleyin
Bash
pip install flask tensorflow scikit-learn pandas numpy
3. Uygulamayı Başlatın
Bash
python app.py
Uygulama başladıktan sonra terminalde çıkan adrese (genellikle https://www.google.com/url?sa=E&source=gmail&q=http://127.0.0.1:5000) giderek web arayüzü üzerinden kod analizi yapmaya başlayabilirsiniz.

🎯 Hedefler
Yapay zeka tarafından üretilen kod içeriklerinin otomatik tespiti.

Farklı NLP modellerinin kod analizi üzerindeki başarı oranlarının karşılaştırılması.

Kodun semantik yapısını anlama becerisine sahip modellerin geliştirilmesi.

👩‍💻 Yazar
Ceren Serbest
