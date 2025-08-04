# train.py

import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from trip_data import TripletFaceDataset
from loss import TripletLoss
from model import StrongerFaceModel

# === Параметры ===
BATCH_SIZE = 16
EPOCHS = 20
MARGIN = 1.0
LEARNING_RATE = 1e-3
DATASET_DIR = 'processed_dataset/train/'
MODEL_SAVE_PATH = 'face_model.pth'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Аугментации ===
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# === Датасет и DataLoader ===
dataset = TripletFaceDataset(root_dir=DATASET_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# === Модель, функция потерь, оптимизатор ===
model = StrongerFaceModel(embedding_size=128).to(DEVICE)
criterion = TripletLoss(margin=MARGIN)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === Тренировка ===
loss_history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{EPOCHS}")

    for i, (anchor, positive, negative) in loop:
        anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)

        optimizer.zero_grad()
        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)

        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(dataloader)
    loss_history.append(avg_loss)
    print(f"[Epoch {epoch}] Average Loss: {avg_loss:.4f}")

    # Сохраняем каждые 5 эпох
    if epoch % 5 == 0:
        torch.save(model.state_dict(), f"face_model_epoch_{epoch}.pth")

# === Финальное сохранение ===
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Тренировка завершена. Модель сохранена в {MODEL_SAVE_PATH}")

# === Визуализация потерь ===
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS + 1), loss_history, marker='o')
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("training_loss.png")
plt.show()
