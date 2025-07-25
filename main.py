import cv2
import os
import numpy as np
from datetime import datetime
# Папка для сохранения снимков

SAVE_DIR = "dataset"
os.makedirs(SAVE_DIR, exist_ok=True)
# Название окна

dummy_window_name = "Нажмите ПРОБЕЛ для снимка (ESC — выход)"
# Пустое изображение-заглушка (чёрное)

blank_image = np.zeros((100, 400, 3), dtype=np.uint8)
cv2.putText(blank_image, "Press SPACE to capture, ESC to quit", (10, 60),
cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

cv2.imshow(dummy_window_name, blank_image)
print("✅ Программа запущена. Окно открыто...")

while True:
    key = cv2.waitKey(1)

    if key == 27:  # ESC
        print("🚪 Выход из программы...")
        break

    if key == 32:  # ПРОБЕЛ
        print("📷 Нажата ПРОБЕЛ. Открытие камеры...")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Камера не найдена.")
            continue

        ret, frame = cap.read()
        cap.release()
        print("✅ Камера закрыта.")

        if not ret:
            print("❌ Ошибка при получении кадра.")
            continue

        # Сохраняем снимок
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.jpg")
        filepath = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(filepath, frame)
        print(f"✅ Снимок сохранён: {filepath}")

cv2.destroyAllWindows()
print("👋 Программа завершена.")
cv2.destroyAllWindows()