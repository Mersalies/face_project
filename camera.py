import cv2
import os
from datetime import datetime



def take_photo():
    """
    Делаем снимок экрана и возрощаем текущую фотографию
    """


    cap = cv2.VideoCapture(0)
    if not cap.isOpened() :
        print("Ошибка: Камера не найдена или не может быть открыта.")
        return 


    success, photo = cap.read() # Читаем кадр из камеры
    cap.release() # Освобождаем ресурсы камеры

    if not success:
        print("Ошибка: Не удалось сделать снимок.")
        return None

    return photo



def take_photo_and_save_photo_to_dir(output_folder: str = "camera_photos", file_prefix: str = "")->str:
    """
    Делает фотографию с веб-камеры и сохраняет ее в указанную папку.

    Args:
        output_folder (str): Папка, куда будет сохранен снимок.
                             Если папка не существует, она будет создана.
                             По умолчанию: "camera_photos".
        file_prefix (str): Префикс для имени файла фотографии. Например, "my_face_".
                           По умолчанию: "" (нет префикса).

    Returns:
        str or None: Полный путь к сохраненному файлу фотографии, если снимок сделан успешно,
                     иначе None.
    """


    cap = cv2.VideoCapture(0)
    if not cap.isOpened() :
        print("Ошибка: Камера не найдена или не может быть открыта.")
        return 


    success, photo = cap.read() # Читаем кадр из камеры
    cap.release() # Освобождаем ресурсы камеры

    if not success:
        print("Ошибка: Не удалось сделать снимок.")
        return None


    # Генерируем имя файла с временной меткой
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if file_prefix:
        file_name = f"{file_prefix}{timestamp}.jpg"
    else:
        file_name = f"{timestamp}.jpg"

    file_path = os.path.join(output_folder, file_name)

    cv2.imwrite(file_path, photo) # Сохраняем фотографию
    print(f"Фотография сохранена: {file_path}")

    return file_path