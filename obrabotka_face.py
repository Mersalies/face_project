import os
import cv2
import torch

from PIL import Image
from tqdm import tqdm

from torchvision import transforms
from facenet_pytorch import MTCNN
from torchvision.transforms.functional import to_pil_image

from model import StrongerFaceModel  # Убедись, что путь корректный



# Устройство: CUDA или CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Инициализируем MTCNN
mtcnn = MTCNN(image_size=160, margin=20, device=device)


# Загружаем твою модель
# model = StrongerFaceModel(embedding_size=128).to(device)
# model.load_state_dict(torch.load("face_model.pth", map_location=device))
# model.eval()




#                              ==== Обработка датасета ====


# RAW_DIR = "archive/train"                # папка с оригинальными изображениями
# OUT_DIR = "processed_dataset/train"      # папка с обработаными изображениями
# def process_face_dataset_emd(raw_dir: str, out_dir: str, image_size: int = 160, margin: int = 20):
#     """
#     Обрабатывает набор фотографий лиц, находя лица, обрезает их
#     и сохраняет в новую директорию.

#     Args:
#         raw_dir (str): Путь к корневой папке с исходными изображениями.
#                        Ожидается структура: raw_dir/person_name/img.jpg
#         out_dir (str): Путь к корневой папке для сохранения обработанных изображений.
#                        Будет создана структура: out_dir/person_name/img.jpg
#         image_size (int): Размер (сторона квадрата) обрезанного лица в пикселях. По умолчанию 160.
#         margin (int): Отступ в пикселях вокруг лица при обрезке. По умолчанию 20.
#     """

#     # Создаём выходную папку, если её нет
#     os.makedirs(out_dir, exist_ok=True)

#     # Включаем использование CUDA если оно есть, иначе CPU
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Используется устройство: {device}")

#     # Инициализируем MTCNN
#     mtcnn = MTCNN(image_size=image_size, margin=margin, device=device)
#     print(f"MTCNN инициализирован с image_size={image_size}, margin={margin}.")

#     # Проходим по каждой папке (каждый человек)
#     for person in os.listdir(raw_dir):
#         person_dir = os.path.join(raw_dir, person)

#         if not os.path.isdir(person_dir):
#             print(f"Пропускаем {person_dir}, так как это не директория.")
#             continue

#         out_person_dir = os.path.join(out_dir, person)
#         os.makedirs(out_person_dir, exist_ok=True)
#         print(f"Обработка изображений для: {person}...")

#         image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#         if not image_files:
#             print(f"В папке {person_dir} не найдено изображений. Пропускаем.")
#             continue

#         for img_name in tqdm(image_files, desc=f"Прогресс для {person}"):
#             img_path = os.path.join(person_dir, img_name)
#             out_path = os.path.join(out_person_dir, img_name)

#             try:
#                 img = Image.open(img_path).convert('RGB')
#                 face = mtcnn(img)

#                 if face is not None:
#                     to_pil_image(face).save(out_path)

#             except Exception as e:
#                 print(f"Ошибка при обработке {img_path}: {e}")

#     print("Обработка завершена!")







#                            ==== Обработка фотографии в эмбеддинг ====

# def get_face_embedding(img: Image.Image) -> torch.Tensor:
#     """
#     Получает эмбеддинг лица из PIL-изображения.

#     Args:
#         img (PIL.Image): Изображение.

#     Returns:
#         torch.Tensor: Эмбеддинг размерности (1, 128) или None, если лицо не найдено.
#     """


#     # Извлечение лица
#     face = mtcnn(img)
#     if face is None:
#         print("❌ Лицо не найдено.")
#         return None

#     # Приведение к формату батча
#     face = face.unsqueeze(0).to(device)

#     # Прогон через модель
#     with torch.no_grad():
#         embedding = model(face)



#     print("⏺️ Извлечено лицо:", face.shape)

#     return embedding  # torch.Size([1, 128])




#                       ==== Обработанные фотографии в эмбеддинг ====

# def generation_embedding(raw_dir: str, out_dir: str, embedding_size: int = 128):
#     """
#     Обрабатывает все изображения в папке и сохраняет эмбеддинги.

#     Args:
#         input_dir (str): Путь к папке с лицами (160x160, RGB).
#         output_dir (str): Куда сохранить эмбеддинги.
#         embedding_size (int): Размер эмбеддинга, по умолчанию 128.
#     """



#     os.makedirs(out_dir, exist_ok=True)
#     transform = transforms.ToTensor()


#     for filename in os.listdir(raw_dir):
#         if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
#             continue

#         image_path = os.path.join(raw_dir, filename)
#         image = Image.open(image_path).convert("RGB")
#         tensor = transform(image).unsqueeze(0).to(device)

#         with torch.no_grad():
#             embedding = model(tensor)

#         name, _ = os.path.splitext(filename)
#         save_path = os.path.join(out_dir, f"{name}.pt")
#         torch.save(embedding.cpu(), save_path)

#         print(f"✅ Эмбеддинг сохранён: {save_path}")




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

    # Инициализируем MTCNN внутри функции
    mtcnn = MTCNN(image_size=image_size, margin=margin, device=device)
    print(f"MTCNN инициализирован с image_size={image_size}, margin={margin}.")

    # Основной цикл, в котором проходимся по папкам с именами людей
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
