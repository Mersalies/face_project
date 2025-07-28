import torch
import torch.nn.functional as F
import os
from torchvision import transforms
from PIL import Image
from model import Face_model
from obrabotka_face import process_two_faces  
from camera import take_photo_from_camera


# === Устройство и модель ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Face_model()
model.load_state_dict(torch.load("face_model.pth", map_location=device))
model.eval().to(device)

# === Трансформация ===
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Предобработка одного изображения ===
def preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    return image

# === Получение эмбеддинга ===
def get_embedding(img_path):
    image = preprocess_image(img_path)
    with torch.no_grad():
        embedding = model(image)
    return embedding.squeeze(0)

# === Косинусная дистанция ===
def cosine_distance(emb1, emb2):
    return 1 - F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

# === Основной блок ===
photo_1 = 'my_photo/test/2025-07-28_23-32-42.jpg'
photo_2 = take_photo_from_camera
processed_dir = 'my_photo/emd'

processed_paths = process_two_faces(photo_1, photo_2, processed_dir)

if None not in processed_paths:
    emb1 = get_embedding(processed_paths[0])
    emb2 = get_embedding(processed_paths[1])

    distance = cosine_distance(emb1, emb2)

    threshold = 0.4
    if distance < threshold:
        print("✅ Это вы (совпадение)")
    else:
        print("❌ Это не вы (не совпадает)")

    print(f"Косинусная дистанция: {distance:.4f}")
else:
    print("⛔ Сравнение невозможно: не удалось найти лицо на одном из изображений.")
