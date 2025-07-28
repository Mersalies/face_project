import torch
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import Face_model

# Настройки
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'models/face_model.pth'
val_dir = 'archive/val'  # структура: val/person1/img1.jpg, ...

# 1. Трансформации (такие же как при обучении)
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 2. Загрузка модели
model = Face_model(embedding_size=128)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# 3. Загружаем валидационные данные (ImageFolder)
dataset = datasets.ImageFolder(val_dir, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# 4. Собираем эмбеддинги
embeddings = []
labels = []

with torch.no_grad():
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        embs = model(imgs)
        embeddings.append(embs.cpu())
        labels.append(lbls)

embeddings = torch.cat(embeddings)
labels = torch.cat(labels)

# 5. Считаем расстояния и точность (по принципу kNN, k=1)
def accuracy(embs, lbls):
    embs = F.normalize(embs)  # нормализация для cosine similarity
    correct = 0
    total = len(embs)

    for i in range(total):
        query = embs[i]
        true_label = lbls[i]

        others = torch.cat((embs[:i], embs[i+1:]))
        other_labels = torch.cat((lbls[:i], lbls[i+1:]))

        # Косинусное расстояние
        dists = F.cosine_similarity(query.unsqueeze(0), others)
        pred_idx = torch.argmax(dists)
        pred_label = other_labels[pred_idx]

        if pred_label == true_label:
            correct += 1

    return correct / total

acc = accuracy(embeddings, labels)
print(f'🔍 Accuracy (cosine kNN = 1): {acc * 100:.2f}%')
