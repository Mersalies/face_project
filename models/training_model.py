import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from triplet.triplet_dataset import TripletFaceDataset
from triplet.loss import TripletLoss
from models.face_model import Face_model
import os
from tqdm import tqdm

# Параметры
BATCH_SIZE = 32
EPOCHS = 20
MARGIN = 1.0
LEARNING_RATE = 1e-3
DATASET_DIR = 'processed_dataset'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = 'models/face_model.pth'

# Аугментации
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
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

    # Сохраняем модель каждый 5 эпох
    if epoch % 5 == 0:
        torch.save(model.state_dict(), f"models/face_model_epoch_{epoch}.pth")

# Финальное сохранение модели
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Тренировка завершена. Модель сохранена в {MODEL_SAVE_PATH}")
