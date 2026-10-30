# Crop recommendation – ben041/Soil-Crop-Recommendation-Models (same backend as disease)
# Features: N, P, K, temperature, humidity, ph, rainfall

import os
import joblib
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(__file__), "Models")
CROP_MODEL_FILE = os.path.join(MODEL_DIR, "RandomForest.pkl")
CROP_LABEL_MAP_FILE = os.path.join(MODEL_DIR, "label_map.pkl")

_crop_model = None
_crop_reverse_map = None


def _load_crop_model():
    global _crop_model, _crop_reverse_map
    if not os.path.isfile(CROP_MODEL_FILE):
        raise FileNotFoundError(
            f"Crop model not found: {CROP_MODEL_FILE}. Clone https://github.com/ben041/Soil-Crop-Recommendation-Models "
            "and copy Advance_models/RandomForest.pkl and Advance_models/label_map.pkl into temp_disease_model/Models/"
        )
    if not os.path.isfile(CROP_LABEL_MAP_FILE):
        raise FileNotFoundError(
            f"Crop label map not found: {CROP_LABEL_MAP_FILE}. Copy Advance_models/label_map.pkl into temp_disease_model/Models/"
        )
    _crop_model = joblib.load(CROP_MODEL_FILE)
    label_map = joblib.load(CROP_LABEL_MAP_FILE)
    _crop_reverse_map = {v: k for k, v in label_map.items()}
    return _crop_model, _crop_reverse_map


def predict_crop(N: float, P: float, K: float, temperature: float, humidity: float, ph: float, rainfall: float) -> str:
    """Predict single recommended crop. Returns crop name."""
    top = predict_crop_top_k(N, P, K, temperature, humidity, ph, rainfall, k=1)
    return top[0][0] if top else "Unknown"


def predict_crop_top_k(
    N: float, P: float, K: float, temperature: float, humidity: float, ph: float, rainfall: float, k: int = 5
) -> list[tuple[str, float]]:
    """Predict top-k recommended crops with confidence (probability). Returns [(crop_name, confidence), ...]."""
    global _crop_model, _crop_reverse_map
    if _crop_model is None:
        _load_crop_model()
    df = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
    )
    # Random Forest has predict_proba: [n_samples, n_classes], we have 1 sample
    proba = _crop_model.predict_proba(df)[0]
    classes = _crop_model.classes_  # class indices
    indexed = [(int(classes[i]), float(proba[i])) for i in range(len(classes))]
    indexed.sort(key=lambda x: -x[1])
    result = []
    for label, conf in indexed[:k]:
        name = _crop_reverse_map.get(label, "Unknown")
        if name and name != "Unknown":
            result.append((name, round(conf, 4)))
    return result
