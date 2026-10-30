# Plant_AI model loader – https://github.com/soumyajit4419/Plant_AI
# ResNet34, 38 classes (New Plant Diseases Dataset)

import io
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]

transform = transforms.Compose([
    transforms.Resize(size=(128, 128)),
    transforms.ToTensor(),
])


class Plant_Disease_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = models.resnet34(weights=None)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 38)

    def forward(self, xb):
        return self.network(xb)


def _load_model():
    import os
    path = os.path.join(os.path.dirname(__file__), "Models", "plantDisease-resnet34.pth")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Model file not found: {path}. Download plantDisease-resnet34.pth from "
            "https://github.com/soumyajit4419/Plant_AI (Models folder) and place it in temp_disease_model/Models/"
        )
    model = Plant_Disease_Model()
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model


_model = None


def get_model():
    global _model
    if _model is None:
        _model = _load_model()
    return _model


def is_likely_leaf(img_bytes):
    """
    Heuristic check: reject images that clearly are not close-up leaf/plant
    (e.g. landscape with sky, people in field). Returns True only if image looks like a leaf.
    """
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return False
    w, h = img.size
    if w < 8 or h < 8:
        return False
    # Very wide or very tall images are usually landscapes/panoramas, not leaf close-ups
    if max(w, h) / min(w, h) > 2.2:
        return False
    # Use fixed size for consistent checks
    img = img.resize((64, 64), Image.Resampling.LANCZOS)
    pixels = list(img.getdata())
    n = len(pixels)
    if n == 0:
        return False
    mean_r = sum(p[0] for p in pixels) / n
    mean_g = sum(p[1] for p in pixels) / n
    mean_b = sum(p[2] for p in pixels) / n
    # Leaf close-ups are green/brown/yellow dominant; very little blue
    if mean_b > mean_g:
        return False
    if mean_b > 95:
        return False
    # Significant sky: any notable "sky blue" region
    sky_blue = sum(1 for r, g, b in pixels if b > 150 and r < 150 and g < 150)
    if sky_blue / n > 0.06:
        return False
    # Top third often has sky in landscapes; if blue dominates there, reject
    top_n = 64 * 21
    top_pixels = pixels[:top_n]
    top_b = sum(p[2] for p in top_pixels) / top_n
    top_g = sum(p[1] for p in top_pixels) / top_n
    if top_b > 100 and top_b > top_g:
        return False
    # Green should dominate for plant/leaf (overall)
    if mean_g < 60 and mean_r < 60:
        return False
    return True


def predict_image(img_bytes):
    """Predict class name and confidence from image bytes. Returns (class_name, confidence)."""
    model = get_model()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    tensor = transform(img)
    xb = tensor.unsqueeze(0)
    with torch.no_grad():
        logits = model(xb)
    probs = torch.softmax(logits, dim=1)
    confidence, idx = torch.max(probs, dim=1)
    class_name = CLASSES[idx[0].item()]
    conf = float(confidence[0].item())
    return class_name, conf
