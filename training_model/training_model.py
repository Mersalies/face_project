import torch
import os

from torch.utils.data import DataLoader
from torchvision import transforms


from training_model.trip_data import TripletFaceDataset
from training_model.loss import TripletLoss
from model import Face_model

from tqdm import tqdm


# Параметры
BATCH_SIZE = 24
EPOCHS = 20
MARGIN = 1.0
LEARNING_RATE = 1e-3
DATASET_DIR = 'processed_dataset/train/' # Это будет работать, так как processed_dataset тоже прямая подпапка
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = 'models/face_model.pth' # Путь для сохранения модели также верен относительно текущего скрипта

# Аугментации
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Исправленная нормализация
])

# Датасет и DataLoader
dataset = TripletFaceDataset(root_dir=DATASET_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# Модель и оптимизатор
model = Face_model(embedding_size=128).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = TripletLoss(margin=MARGIN)

# Цикл обучения
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
    print(f"[Epoch {epoch}/{EPOCHS}] Average Loss: {avg_loss:.4f}")

    # Сохраняем модель каждые 5 эпох
    if epoch % 5 == 0:
        torch.save(model.state_dict(), f"models/face_model_epoch_{epoch}.pth")

# Финальное сохранение модели
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Тренировка завершена. Модель сохранена в {MODEL_SAVE_PATH}")