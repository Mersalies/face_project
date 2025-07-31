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


def process_two_faces(img_path1: str, img_path2: str, output_dir: str, image_size: int = 160, margin: int = 20) -> List[Optional[str]]:
    """
    Извлекает и сохраняет лица с двух предоставленных изображений.

    Эта функция использует MTCNN для обнаружения лиц на двух изображениях, 
    изменяет их размер до 160x160 пикселей (стандартный размер для многих моделей
    распознавания лиц, таких как FaceNet) и сохраняет их в указанной директории.
    Если лицо не найдено или произошла ошибка, соответствующий элемент в списке
    результатов будет равен None.

    Args:
        img_path1 (str): Путь к первому файлу изображения.
        img_path2 (str): Путь ко второму файлу изображения.
        output_dir (str): Путь к директории для сохранения обработанных лиц.
        image_size (int): Размер (сторона квадрата) для сохранения лица в пикселях. 
                          По умолчанию 160.
        margin (int): Отступ в пикселях вокруг найденного лица. По умолчанию 20.

    Returns:
        List[Optional[str]]: Список, содержащий пути к сохраненным файлам лиц.
                           Каждый элемент списка - это путь к файлу или None, 
                           если лицо не было найдено.
    """
    # Создаем выходную директорию, если она не существует
    os.makedirs(output_dir, exist_ok=True)
    
    # Определяем доступное устройство для ускорения обработки (GPU/CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используется устройство: {device}")

    # Инициализируем детектор лиц MTCNN с заданными параметрами
    # MTCNN (Multi-task Cascaded Convolutional Networks) - это эффективный
    # фреймворк для обнаружения лиц.
    mtcnn = MTCNN(image_size=image_size, margin=margin, device=device)

    output_paths = []
    # Итерируемся по двум путям к изображениям для последовательной обработки
    for idx, path in enumerate([img_path1, img_path2], start=1):
        try:
            # Загружаем изображение и конвертируем его в формат RGB
            img = Image.open(path).convert('RGB')
            
            # Обнаруживаем лицо на изображении.
            # Если лицо найдено, `mtcnn` вернет тензор, представляющий обрезанное лицо.
            face = mtcnn(img)
            if face is not None:
                # Генерируем уникальное имя файла для сохранения
                output_path = os.path.join(output_dir, f"face_{idx}.jpg")
                
                # Конвертируем тензор лица обратно в изображение PIL и сохраняем его
                to_pil_image(face).save(output_path)
                output_paths.append(output_path)
                print(f"Лицо успешно сохранено: {output_path}")
            else:
                print(f"❌ Лицо не найдено на изображении: {path}")
                output_paths.append(None)
        except Exception as e:
            # Обработка возможных ошибок, таких как поврежденный файл или некорректный путь
            print(f"❌ Ошибка при обработке изображения {path}: {e}")
            output_paths.append(None)

    return output_paths
