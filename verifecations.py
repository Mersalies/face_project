import torch
import torch.nn.functional as F
import os
from torchvision import transforms
from PIL import Image
from model import Face_model 
from obrabotka_face import process_two_faces 
from camera import take_photo_from_camera #



# --- Устройство и Модель ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Face_model()
try:
    model.load_state_dict(torch.load("face_model.pth", map_location=device))
    model.eval().to(device)
except FileNotFoundError:
    print("Ошибка: Файл 'face_model.pth' не найден. Убедитесь, что модель обучена и сохранена.")
    exit() # Завершаем выполнение, если модель не найдена




# --- Трансформация ---
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Исправленная нормализация
])




# --- Предобработка Одного Изображения ---
def preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    return image




# --- Получение Эмбеддинга ---
def get_embedding(img_path):
    image = preprocess_image(img_path)
    with torch.no_grad():
        embedding = model(image)
    return embedding.squeeze(0)



# --- Косинусная Дистанция ---
def cosine_distance(emb1, emb2):
    return 1 - F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()



# --- Основной Блок ---
photo_1 = 'my_photo/test/photo_2025-07-29_00-41-53.jpg'

camera_photos_folder = 'my_photo/camera_captures' # Папка для сохранения снимков с камеры
photo_2_path = take_photo_from_camera(output_folder=camera_photos_folder) # Вызываем функцию для снимка

processed_dir = 'my_photo/emd' # Папка для обработанных лиц

if photo_2_path: # Проверяем, что снимок с камеры успешно сделан
    processed_paths = process_two_faces(photo_1, photo_2_path, processed_dir)

    if None not in processed_paths:
        emb1 = get_embedding(processed_paths[0])
        emb2 = get_embedding(processed_paths[-1])

        distance = cosine_distance(emb1, emb2)

        threshold = 0.49 # Порог для сравнения
        if distance < threshold:
            print("✅ Это вы (совпадение)")
        else:
            print("❌ Это не вы (не совпадает)")

        print(f"Косинусная дистанция: {distance:.4f}")
    else:
        print("⛔ Сравнение невозможно: не удалось найти лицо на одном из изображений.")
else:
    print("⛔ Сравнение невозможно: не удалось сделать снимок с камеры.")