import cv2
import os
from datetime import datetime


papka = "dataset"
os.makedirs(papka,exist_ok=True)


while True:

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Камера не найдена")
        continue

    succses,photo = cap.read()
    cap.release()
    if not succses:
        print("Ошибка снимка")
        continue

    file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.jpg")
    file_path = os.path.join(papka, file_name)

    cv2.imwrite(file_path, photo)
    break


cv2.destroyAllWindows()
print("Программа завершена.")



