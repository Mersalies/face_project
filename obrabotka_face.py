from PIL import Image
from facenet_pytorch import MTCNN
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

import torch
import os


# RAW_DIR = "archive/train"                # папка с оригинальными изображениями
# OUT_DIR = "processed_dataset/train"      # папка с обработаными изображениями

def process_face_dataset(raw_dir: str, out_dir: str, image_size: int = 160, margin: int = 20):
    """
    Обрабатывает набор фотографий лиц, находя лица, обрезает их
    и сохраняет в новую директорию.

    Args:
        raw_dir (str): Путь к корневой папке с исходными изображениями.
                       Ожидается структура: raw_dir/person_name/img.jpg
        out_dir (str): Путь к корневой папке для сохранения обработанных изображений.
                       Будет создана структура: out_dir/person_name/img.jpg
        image_size (int): Размер (сторона квадрата) обрезанного лица в пикселях. По умолчанию 160.
        margin (int): Отступ в пикселях вокруг лица при обрезке. По умолчанию 20.
    """

    # Создаём выходную папку, если её нет
    os.makedirs(out_dir, exist_ok=True)

    # Включаем использование CUDA если оно есть, иначе CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используется устройство: {device}")

    # Инициализируем MTCNN
    mtcnn = MTCNN(image_size=image_size, margin=margin, device=device)
    print(f"MTCNN инициализирован с image_size={image_size}, margin={margin}.")

    # Проходим по каждой папке (каждый человек)
    for person in os.listdir(raw_dir):
        person_dir = os.path.join(raw_dir, person)

        if not os.path.isdir(person_dir):
            print(f"Пропускаем {person_dir}, так как это не директория.")
            continue

        out_person_dir = os.path.join(out_dir, person)
        os.makedirs(out_person_dir, exist_ok=True)
        print(f"Обработка изображений для: {person}...")

        image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f"В папке {person_dir} не найдено изображений. Пропускаем.")
            continue

        for img_name in tqdm(image_files, desc=f"Прогресс для {person}"):
            img_path = os.path.join(person_dir, img_name)
            out_path = os.path.join(out_person_dir, img_name)

            try:
                img = Image.open(img_path).convert('RGB')
                face = mtcnn(img)

                if face is not None:
                    to_pil_image(face).save(out_path)

            except Exception as e:
                print(f"Ошибка при обработке {img_path}: {e}")

    print("Обработка завершена!")


def process_two_faces(img_path1: str, img_path2: str, output_dir: str, image_size: int = 160, margin: int = 20):
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используется устройство: {device}")

    mtcnn = MTCNN(image_size=image_size, margin=margin, device=device)

    output_paths = []
    for idx, path in enumerate([img_path1, img_path2], start=1):
        try:
            img = Image.open(path).convert('RGB')
            face = mtcnn(img)
            if face is not None:
                output_path = os.path.join(output_dir, f"face_{idx}.jpg")
                to_pil_image(face).save(output_path)
                output_paths.append(output_path)
                print(f"Лицо сохранено: {output_path}")
            else:
                print(f"❌ Лицо не найдено на изображении: {path}")
                output_paths.append(None)
        except Exception as e:
            print(f"Ошибка при обработке {path}: {e}")
            output_paths.append(None)

    return output_paths
