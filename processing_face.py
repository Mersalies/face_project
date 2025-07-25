import os
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm

import torch as th


RAW_DIR = "archive/train"
OUT_DIR = "processed_dataset/train"


# Создаём выходную папку
os.makedirs(OUT_DIR, exist_ok=True)

# включаем использование cuda если оно есть
device = 'cuda' if th.cuda.is_available() else 'cpu'


mtcnn = MTCNN(image_size=160, margin=20, device=device)

