import torch
import torch.nn.functional as F
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Face_model
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

# === Настройки ===
VAL_DIR = 'processed_dataset/val'
MODEL_PATH = 'face_model.pth'
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_SIZE = 128

# === Параметры для семплирования ===
MAX_ROC_PAIRS = 15_000_000

# === Трансформации ===
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Загрузка валидационного датасета ===
print(f"Загрузка датасета из: {VAL_DIR}")
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Датасет загружен. Найдено {len(val_dataset)} изображений в {len(val_loader)} батчах.")

# === Загрузка модели ===
print(f"Загрузка модели из: {MODEL_PATH} на {DEVICE}...")
model = Face_model(embedding_size=EMBEDDING_SIZE).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Модель успешно загружена.")

# === Извлечение эмбеддингов ===
print("Извлечение эмбеддингов из изображений...")
start_time_embedding = time.time()
embeddings = []
labels = []

with torch.no_grad():
    for imgs, lbls in tqdm(val_loader, desc="Извлечение эмбеддингов"):
        imgs = imgs.to(DEVICE)
        embs = model(imgs)
        embeddings.append(embs.cpu())
        labels.append(lbls)

embeddings = torch.cat(embeddings)
labels = torch.cat(labels)
end_time_embedding = time.time()
print(f"Извлечено {embeddings.shape[0]} эмбеддингов за {end_time_embedding - start_time_embedding:.2f} секунд.")

# === Улучшенная функция для подсчета расстояний (с разбиением на чанки) ===
def calculate_all_pairwise_distances_and_labels_chunked(embs, lbls, distance_type='cosine', chunk_size=1024):
    N = embs.size(0)
    print(f"Начало вычисления {N*(N-1)//2} попарных расстояний с использованием чанков по {chunk_size}...")

    all_distances_list = []
    all_pair_labels_list = []

    if distance_type == 'cosine':
        embs = F.normalize(embs, p=2, dim=1)

    for i in tqdm(range(0, N, chunk_size), desc="Вычисление попарных расстояний"):
        chunk_embs = embs[i:min(i + chunk_size, N)].to(DEVICE)
        
        if distance_type == 'cosine':
            similarity_chunk = torch.matmul(chunk_embs, embs.transpose(0, 1).to(DEVICE))
            distance_chunk = 1 - similarity_chunk
        elif distance_type == 'euclidean':
            distance_chunk = torch.cdist(chunk_embs, embs.to(DEVICE), p=2)
        else:
            raise ValueError("Unknown distance type. Choose 'cosine' or 'euclidean'.")

        for idx_in_chunk in range(chunk_embs.size(0)):
            current_global_idx = i + idx_in_chunk
            
            dists_from_current = distance_chunk[idx_in_chunk, current_global_idx + 1:]
            labels_from_current = (lbls[current_global_idx] == lbls[current_global_idx + 1:])

            all_distances_list.append(dists_from_current.cpu())
            all_pair_labels_list.append(labels_from_current.cpu())
        
        del chunk_embs
        if distance_type == 'cosine':
            del similarity_chunk
        else:
            del distance_chunk
        torch.cuda.empty_cache()

    all_distances_np = torch.cat(all_distances_list).numpy()
    all_pair_labels_np = torch.cat(all_pair_labels_list).numpy()
    
    print("Вычисление попарных расстояний завершено.")
    return all_distances_np, all_pair_labels_np


# === Расчёт расстояний ===
start_time_distances = time.time()
all_distances_raw, true_labels_raw = calculate_all_pairwise_distances_and_labels_chunked(embeddings, labels, distance_type='cosine', chunk_size=1024)
end_time_distances = time.time()
print(f"Попарные расстояния вычислены за {end_time_distances - start_time_distances:.2f} секунд.")

# === Семплирование для ROC-кривой и метрик ===
print(f"\nОбщее количество пар: {len(all_distances_raw)}. Семплирование до {MAX_ROC_PAIRS} пар для оценки метрик...")

if len(all_distances_raw) > MAX_ROC_PAIRS:
    sample_indices = np.random.choice(len(all_distances_raw), MAX_ROC_PAIRS, replace=False)
    
    all_distances = all_distances_raw[sample_indices]
    true_labels = true_labels_raw[sample_indices]
    print(f"Выбрано {len(all_distances)} случайных пар.")
else:
    all_distances = all_distances_raw
    true_labels = true_labels_raw
    print("Количество пар меньше MAX_ROC_PAIRS, используются все пары.")


# Разделение на положительные и отрицательные расстояния для статистики
pos_dists = all_distances[true_labels == 1]
neg_dists = all_distances[true_labels == 0]

print(f"\n✅ Положительных пар (один человек) в выборке: {len(pos_dists)}")
print(f"✅ Отрицательных пар (разные люди) в выборке: {len(neg_dists)}")
print(f"📌 Пример расстояний (pos): {pos_dists[:5]}")
print(f"📌 Пример расстояний (neg): {neg_dists[:5]}")


# === Средние значения ===
avg_pos = np.mean(pos_dists) if len(pos_dists) > 0 else 0
avg_neg = np.mean(neg_dists) if len(neg_dists) > 0 else 0

print(f"\n📊 Средняя внутриклассовая дистанция (один человек): {avg_pos:.4f}")
print(f"📊 Средняя межклассовая дистанция (разные люди):     {avg_neg:.4f}")

# === Accuracy по порогу ===
print("\nВычисление ROC-кривой и точности...")
start_time_metrics = time.time()

fpr, tpr, thresholds = roc_curve(true_labels, -all_distances)
roc_auc = auc(fpr, tpr)

best_idx = np.argmax(tpr - fpr)
best_threshold = -thresholds[best_idx]

preds = (all_distances <= best_threshold).astype(int)
accuracy = np.mean(preds == true_labels)

end_time_metrics = time.time()
print(f"Метрики вычислены за {end_time_metrics - start_time_metrics:.2f} секунд.")

print(f"\n✅ Accuracy на валидации (по лучшему порогу): {accuracy:.4f}")
print(f"🎯 Лучший порог: {best_threshold:.4f}")
print(f"🔺 AUC ROC: {roc_auc:.4f}")

# === Построение ROC-графика ===
print("\nПостроение ROC-кривой...")
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC-кривая для верификации лиц (выборка)")
plt.legend(loc="lower right")
plt.grid(True)

# Сохраняем график в файл вместо попытки его отображения.
# Вы можете изменить имя файла и формат по желанию.
output_roc_path = "roc_curve_evaluation.png"
plt.savefig(output_roc_path)
print(f"ROC-кривая сохранена как: {output_roc_path}")

# Убираем plt.show() здесь, так как оно не нужно в неинтерактивной среде
# Если вы запустите код в Jupyter Notebook, можете раскомментировать plt.show()
# plt.show() # Эта строка теперь закомментирована

print("Программа завершена.")