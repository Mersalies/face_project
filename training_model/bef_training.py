import torch
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image
from model import Face_model
from obrabotka_face import process_two_faces
from camera import take_photo_from_camera_2v


# --- Конфигурация Устройства и Загрузка Модели ---
# Определяем устройство для выполнения операций (GPU, если доступно, иначе CPU).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Инициализируем модель для распознавания лиц.
model = Face_model()

try:
    # Загружаем предварительно обученные веса модели.
    # 'map_location' позволяет загрузить модель на нужное устройство.
    model.load_state_dict(torch.load("face_model.pth", map_location=device))
    # Переводим модель в режим оценки (inference) и перемещаем на выбранное устройство.
    model.eval().to(device)
except FileNotFoundError:
    # Обработка случая, если файл модели не найден.
    print("Ошибка: Файл 'face_model.pth' не найден. Убедитесь, что модель обучена и сохранена.")
    exit() # Завершаем выполнение программы.


# --- Определение Трансформаций Изображения ---
# Создаем конвейер трансформаций для предобработки входных изображений.
# 1. Изменение размера до (160, 160) пикселей.
# 2. Преобразование изображения в тензор PyTorch.
# 3. Нормализация пиксельных значений в диапазон [-1, 1] с заданными средним и стандартным отклонением.
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# --- Функция Предобработки Одного Изображения ---
def preprocess_image(img_path):
    """
    Загружает изображение, применяет к нему предопределенные трансформации
    и подготавливает его для подачи в модель.

    Args:
        img_path (str): Путь к изображению.

    Returns:
        torch.Tensor: Предобработанное изображение в виде тензора.
    """
    # Открываем изображение и конвертируем его в формат RGB.
    image = Image.open(img_path).convert("RGB")
    # Применяем трансформации, добавляем размерность батча (unsqueeze)
    # и перемещаем тензор на выбранное устройство.
    image = transform(image).unsqueeze(0).to(device)
    return image


# --- Функция Получения Эмбеддинга ---
def get_embedding(img_path):
    """
    Получает эмбеддинг (вектор признаков) изображения с помощью модели.

    Args:
        img_path (str): Путь к изображению.

    Returns:
        torch.Tensor: Эмбеддинг изображения.
    """
    # Предобрабатываем изображение.
    image = preprocess_image(img_path)
    # Отключаем расчет градиентов для ускорения и уменьшения потребления памяти
    # в режиме оценки.
    with torch.no_grad():
        # Передаем изображение через модель для получения эмбеддинга.
        embedding = model(image)
    # Убираем размерность батча из эмбеддинга.
    return embedding.squeeze(0)


# --- Функция Расчета Косинусного Расстояния ---
def cosine_distance(emb1, emb2):
    """
    Вычисляет косинусное расстояние между двумя эмбеддингами.
    Меньшее значение расстояния указывает на большее сходство.

    Args:
        emb1 (torch.Tensor): Первый эмбеддинг.
        emb2 (torch.Tensor): Второй эмбеддинг.

    Returns:
        float: Значение косинусного расстояния.
    """
    # Вычисляем косинусное сходство и вычитаем его из 1, чтобы получить расстояние.
    # Добавляем размерность батча для каждого эмбеддинга перед расчетом сходства.
    return 1 - F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()


# --- Основной Блок Выполнения Программы ---
if __name__ == "__main__":
    # Путь к первому изображению (например, ваше эталонное фото).
    photo_1 = 'my_photo/test/2025-07-28_23-32-42.jpg'
    # Папка для сохранения фотографий, сделанных с камеры.
    camera_photos_folder = 'my_photo/camera_captures'
    # Папка для сохранения обработанных (обрезанных) лиц.
    processed_dir = 'my_photo/emd'

    print("Делаем снимок с камеры...")
    # Делаем снимок с камеры и получаем путь к сохраненному изображению.
    photo_2_path = take_photo_from_camera_2v(output_folder=camera_photos_folder)

    # Проверяем, был ли снимок с камеры успешно сделан.
    if photo_2_path:
        # Обрабатываем два изображения (находим и обрезаем лица).
        # Возвращает пути к обработанным изображениям лиц.
        processed_paths = process_two_faces(photo_1, photo_2_path, processed_dir)

        # Проверяем, удалось ли найти лица на обоих изображениях.
        if None not in processed_paths:
            # Получаем эмбеддинги для каждого обработанного лица.
            emb1 = get_embedding(processed_paths[0])
            emb2 = get_embedding(processed_paths[-1])

            # Вычисляем косинусное расстояние между эмбеддингами.
            distance = cosine_distance(emb1, emb2)

            # Определяем порог для сравнения лиц.
            # Значения расстояния ниже этого порога считаются совпадением.
            threshold = 0.4
            if distance < threshold:
                print("✅ Это вы (совпадение)")
            else:
                print("❌ Это не вы (не совпадает)")

            # Выводим рассчитанное косинусное расстояние.
            print(f"Косинусная дистанция: {distance:.4f}")
        else:
            print("⛔ Сравнение невозможно: не удалось найти лицо на одном из изображений.")
    else:
        print("⛔ Сравнение невозможно: не удалось сделать снимок с камеры.")