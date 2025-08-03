import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from trip_data import TripletFaceDataset
from loss import TripletLoss
from model import StrongerFaceModel
import os

# === Параметры ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 5
MARGIN = 1.0
LEARNING_RATE = 1e-4

# === Пути ===
dataset_path = 'your_dataset'  # сюда положи свои фото + другие классы
model_path = 'face_model.pth'

# === Трансформации ===
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Исправленная нормализация
])

# === Датасет и загрузчик ===
train_dataset = TripletFaceDataset(dataset_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Модель и оптимизатор ===
model = StrongerFaceModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = TripletLoss(margin=MARGIN)

# === Обучение ===
for epoch in range(EPOCHS):
    total_loss = 0.0
    for batch in train_loader:
        anchor, positive, negative = batch
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        emb_anchor = model(anchor)
        emb_positive = model(positive)
        emb_negative = model(negative)

        loss = criterion(emb_anchor, emb_positive, emb_negative)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Эпоха {epoch+1}/{EPOCHS}, Потери: {avg_loss:.4f}")

# === Сохранение модели ===
torch.save(model.state_dict(), 'face_model_finetuned.pth')
print("✅ Дообученная модель сохранена в face_model_finetuned.pth")
