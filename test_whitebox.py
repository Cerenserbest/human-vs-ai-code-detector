import unittest
from app import app   # Flask uygulamanı buradan import ediyoruz


class TestPredictAPI(unittest.TestCase):
    def setUp(self):
        # Her testten önce Flask test client oluşturuyoruz
        self.client = app.test_client()

    # ✅ WHITE BOX TEST 1
    def test_predict_valid_code_returns_json(self):
        """
        Geçerli bir Python kodu gönderildiğinde
        /predict endpoint'inin 200 ve düzgün JSON döndürmesini test eder.
        """
        payload = {"code": "print('white box test 1')"}

        response = self.client.post("/predict", json=payload)

        # Status 200 mü?
        self.assertEqual(response.status_code, 200)

        data = response.get_json()

        # JSON içinde olması gereken alanlar var mı?
        self.assertIn("final_label", data)
        self.assertIn("final_probability", data)
        self.assertIn("model_1_lr_proba", data)
        self.assertIn("model_2_lstm_proba", data)
        self.assertIn("model_3_gru_proba", data)

    # ✅ WHITE BOX TEST 2
    def test_predict_handles_long_code(self):
        """
        60+ satırlık uzun bir kodu backend'in hata vermeden
        işleyebildiğini test eder.
        """
        long_code = "\n".join([f"print({i})" for i in range(80)])

        response = self.client.post("/predict", json={"code": long_code})

        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        # Yine JSON alanları var mı?
        self.assertIn("final_label", data)
        self.assertIn("final_probability", data)

    # ✅ WHITE BOX TEST 3
    def test_predict_rejects_missing_code_field(self):
        """
        'code' alanı olmayan bir istek atıldığında
        API'nin düzgün hata yönetimi yapıp yapmadığını test eder.
        """
        # Bilerek 'code' göndermiyoruz
        response = self.client.post("/predict", json={})

        # Uygulamana göre 400 ya da 200 dönebilir.
        # Eğer şu an 500 veriyorsa, bunu görüp loguna yazabilmen bile
        # white-box test açısından değerli.
        self.assertNotEqual(response.status_code, 500)


if __name__ == "__main__":
    unittest.main()
# 