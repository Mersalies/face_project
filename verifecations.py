import os
import torch
import torch.nn.functional as F

from camera import take_photo 
from obrabotka_face import get_face_embedding

emd_dir = "photo_to_verifecation/embding_photo"
th = 0.4


# --- Косинусная Дистанция ---
def l2_distance(emb1, emb2):
    emb1 = emb1.view(-1)
    emb2 = emb2.view(-1)
    return torch.norm(emb1 - emb2, p=2).item()


def compare_with_folder(embedding, embeddings_folder, threshold=0.5):
    if embedding is None:
        print("❌ Ошибка: эмбеддинг не получен. Прерывание сравнения.")
        return

    device = embedding.device  # устройство, на котором находится emb1
    matched = False
    count = 0

    for file in os.listdir(embeddings_folder):
        if file.endswith('.pt'):
            emb_path = os.path.join(embeddings_folder, file)
            emb2 = torch.load(emb_path, map_location=device)
            distance = l2_distance(embedding, emb2)

            if distance < threshold:
                count += 1
                matched = True

    if count >= 5:
        print("✅ Это вы (совпадение)")
    else:
        print("⛔ Совпадений не найдено.")


# --- Основной блок ---
Img = take_photo()
emd = get_face_embedding(img=Img)

if emd is None:
    print("❌ Лицо не найдено на фотографии. Повторите попытку.")
else:
    compare_with_folder(embedding=emd, embeddings_folder=emd_dir, threshold=th)
