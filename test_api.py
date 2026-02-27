import requests

URL = "http://127.0.0.1:5000/predict"

test_code = r"""
def validate_and_predict(model, sample):
    \"\"\"
    Validate input shape and type before performing model prediction.
    Ensures numerical consistency and prevents downstream errors.
    \"\"\"

    if not hasattr(model, "predict"):
        raise ValueError("Provided model does not implement predict().")

    if not isinstance(sample, (list, tuple)):
        raise TypeError("Input must be list-like.")

    sample = np.array(sample).reshape(1, -1)
    prediction = model.predict(sample)[0]

    return {"input": sample.tolist(), "prediction": float(prediction)}
"""

resp = requests.post(URL, json={"code": test_code})

print("\n=== Tahmin Sonucu ===")
print(resp.json())
