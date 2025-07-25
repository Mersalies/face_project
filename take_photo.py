import cv2
import os
import numpy as np
from datetime import datetime
import time

# Создание папки dataset и провека на ее наличие
SAVE_DIR = "dataset"
os.makedirs(SAVE_DIR, exist_ok=True)


# Создание заглушки для нажатия на пробел пользователя
# Создаем изображение(матрицу) 200х600, числа > 0
# Текст в нутри изображения с отступом по ширене 30 и высотте 80
matrix_for_cap = np.zeros((200, 600, 3), dtype=np.uint8)
cv2.putText(matrix_for_cap, "Take a photo with SPACE, exit with ESC", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


window_name = "cap"
cv2.imshow("cap", matrix_for_cap)
print(" Программа запущена. Окно открыто и ожидает нажатия...")



# --- Основной цикл программы ---
while True:

    # Ловим нажание клавишы
    key = cv2.waitKey(1)

    if key == 27:  # ловим нажаните - ESC
        print(" Выход из программы по нажатию ESC")
        break

    if key == 32:  # ловим нажаните - SPACE
        print("\n Нажата ПРОБЕЛ. Открытие камеры.")

        # Открытие камеры
        # обработка ошибки открытия камеры 
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():

            print(" Камера не найдена или недоступна.")
            print("Убедитесь, что камера подключена и не используется другим приложением.")
            

            # создание матрицы для экрана ошибки
            # Размещение текста ошибки на изображение
            # создание изображения с текстом ошибки 
            matrix_for_eror = np.zeros((200, 600, 3), dtype=np.uint8)
            cv2.putText(matrix_for_eror, "EROR: cap NOT found", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow(window_name, matrix_for_eror)
            
            time.sleep(1)
            
            cv2.imshow(window_name, matrix_for_cap)
            continue



        # считываем данные с камеры
        # succses - принмает True(если удалось прочитать текущие изображение с камеры) и False(если не удалось прочитать текущие изображение с камеры)
        # photo - будет помещено само изображение        
        succses, photo = cap.read()
        cap.release() # закрытие камеры
        if not succses:

            print(" Error receiving frame from cap.")
            matrix_for_eror = np.zeros((200, 600, 3), dtype=np.uint8)
            cv2.putText(matrix_for_eror, "ERROR: Failed to get frame", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow(window_name, matrix_for_eror)
            
            time.sleep(1)
            cv2.imshow(window_name, matrix_for_cap)
            continue

        # Сохранение захваченного изображения 
        # Создаем название файла по текущей дате
        # указываем путь к папке dataset
        file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.jpg")
        file_path = os.path.join(SAVE_DIR, file_name)

        cv2.imwrite(file_path, photo)

        print(" Снимок сделан. Закрытие программы")
        break 
        

cv2.destroyAllWindows()
print("Программа завершена.")