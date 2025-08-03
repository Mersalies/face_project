import os
import random

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class TripletFaceDataset(Dataset):
    """
    Класс TripletFaceDataset предназначен для загрузки и подготовки данных
    для обучения моделей, использующих триплетную функцию потерь (Triplet Loss).
    Он генерирует триплеты (anchor, positive, negative) изображений "на лету".

    """

    def __init__(self, root_dir, transform=None):
        """
        Инициализация датасета.

        Args:
            root_dir (str): Корневая директория, содержащая папки с изображениями для каждого человека.
                            Ожидается, что каждая подпапка в root_dir является идентификатором человека
                            и содержит его фотографии.
            transform (torchvision.transforms.Compose, optional): Композиция преобразований,
                                                                 применяемых к изображениям.
                                                                 Если None, используются стандартные преобразования:
                                                                 изменение размера до (160, 160),
                                                                 преобразование в тензор и нормализация.
        """
        self.root_dir = root_dir

        # Определяем преобразования изображений.
        # Если пользователь не предоставил свои, используем стандартный набор.
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((160, 160)),  # Изменение размера изображений до 160x160 пикселей.
                transforms.ToTensor(),          # Преобразование PIL Image в PyTorch Tensor.
                # Нормализация тензора изображения. Среднее и стандартное отклонение [0.5, 0.5, 0.5]
                # приводит значения пикселей в диапазон [-1, 1].
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
            ])
        else:
            self.transform = transform

        # --- Сбор информации о данных ---

        # Получаем список всех элементов в корневой директории.
        all_items = os.listdir(root_dir)
        # Фильтруем список, оставляя только директории, которые представляют собой идентификаторы людей.
        self.people = [p for p in all_items if os.path.isdir(os.path.join(root_dir, p))]

        # Создаем словарь, где ключ - это идентификатор человека (имя папки),
        # а значение - список полных путей ко всем его изображениям.
        # Это позволяет быстро получать доступ к фотографиям конкретного человека.
        self.image_paths = {
            person: [
                os.path.join(root_dir, person, img_name)
                for img_name in os.listdir(os.path.join(root_dir, person))
                # Опционально: можно добавить фильтрацию по расширению файла,
                # если в папках могут быть не только изображения.
                # Пример: if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            for person in self.people
        }

        # Отфильтровываем людей, у которых меньше двух фотографий.
        # Для создания триплета нам необходимо минимум два изображения от одного и того же человека
        # (для anchor и positive).
        self.image_paths = {k: v for k, v in self.image_paths.items() if len(v) >= 2}
        # Обновляем список людей, чтобы он соответствовал отфильтрованным данным.
        self.people = list(self.image_paths.keys())

        # Если после фильтрации не осталось людей, это указывает на проблему с данными.
        if not self.people:
            raise ValueError(f"Не найдено достаточное количество людей с >= 2 изображениями в '{root_dir}'. "
                             "Проверьте структуру папок и количество изображений.")

    def __len__(self):
        """
        Возвращает "логический" размер датасета.
        Для триплетной функции потерь триплеты генерируются случайным образом,
        поэтому мы устанавливаем большое фиксированное число.
        Это позволяет DataLoader'у запрашивать большое количество случайных триплетов
        для каждой эпохи обучения, не ограничиваясь конечным количеством предопределенных триплетов.
        """
        return 100000  # Можно настроить это значение в зависимости от желаемого количества шагов за эпоху.

    def __getitem__(self, index):
        """
        Генерирует один триплет изображений (anchor, positive, negative).

        Args:
            index (int): Индекс элемента (игнорируется, так как триплеты генерируются случайно).

        Returns:
            tuple: Кортеж из трех преобразованных тензоров изображений: (anchor, positive, negative).
        """

        # --- Выбираем Anchor и Positive изображения ---
        # Выбираем случайного человека (identity) для anchor и positive изображений.
        person = random.choice(self.people)
        # Выбираем два случайных, но РАЗНЫХ изображения от этого человека.
        # Они будут использоваться как anchor (якорь) и positive (положительный пример).
        imgs = random.sample(self.image_paths[person], 2)
        anchor_path, positive_path = imgs[0], imgs[1]

        # --- Выбираем Negative изображение ---
        # Выбираем случайного человека для negative изображения.
        # Важно: этот человек НЕ ДОЛЖЕН совпадать с 'person' (человеком для anchor/positive).
        # Используем списковое включение для фильтрации.
        negative_person = random.choice([p for p in self.people if p != person])
        # Выбираем случайное изображение от выбранного negative_person.
        negative_path = random.choice(self.image_paths[negative_person])

        # --- Загружаем изображения ---
        # Открываем каждое изображение и конвертируем его в формат RGB
        # (для обеспечения 3-х каналов, даже если исходное изображение в оттенках серого).
        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')

        # --- Применяем преобразования ---
        # Применяем заданные преобразования (изменение размера, нормализация и т.п.)
        # к каждому изображению.
        anchor = self.transform(anchor)
        positive = self.transform(positive)
        negative = self.transform(negative)

        return anchor, positive, negative