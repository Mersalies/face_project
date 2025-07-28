import torch
import torch.nn.functional as F
import cv2
import os
from torchvision import transforms
from PIL import Image
from model import Face_model

# === Загрузка модели ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Face_model()
model.load_state_dict(torch.load("models/face_model.pth", map_location=device))
model.eval().to(device)

# === Трансформации ===
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Функция загрузки и обработки изображения ===
def preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    return image

# === Вычисление эмбеддингов ===
def get_embedding(img_path):
    image = preprocess_image(img_path)
    with torch.no_grad():
        embedding = model(image)
    return embedding.squeeze(0)

# === Расчет расстояния ===
def cosine_distance(emb1, emb2):
    return 1 - F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

# === Основной код ===
img1_path = "your_face.jpg"     # эталонное изображение
img2_path = "test_image.jpg"    # проверяемое изображение

emb1 = get_embedding(img1_path)
emb2 = get_embedding(img2_path)

distance = cosine_distance(emb1, emb2)

# Порог подбирается эмпирически, например 0.5
threshold = 0.5
if distance < threshold:
    print("✅ Это вы (совпадение)")
else:
    print("❌ Это не вы (не совпадает)")

print(f"Косинусная дистанция: {distance:.4f}")
