'''
В данном фале будут единные настройки для модели, тренеровки, оценке и верефикации
'''

from torchvision import transforms

#     ---- Обработка изображений -----
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


#     ---- Обучение модели -----
BATCH_SIZE = 32
VAL_DIR = 'processed_dataset/val'
MODEL_PATH = 'face_model.pth'
EMBEDDING_SIZE = 128










