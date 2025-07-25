import os
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm


import torch as th

RAW_DIR = "archive/train"                # папка с оригинальными изображениями
OUT_DIR = "processed_dataset/train"      # папка с обработаными изображениями


# Создаём выходную папку
os.makedirs(OUT_DIR, exist_ok=True)

# включаем использование cuda если оно есть
device = 'cuda' if th.cuda.is_available() else 'cpu'

# иницеализируем MTCNN для обработки фотографий
# На выходе получим обрезанное лицо 160х160 с отступом от лица в 20 пиксилей
mtcnn = MTCNN(image_size=160, margin=20, device=device)

# основной цикл в котором проходимся по папкам 
for person in os.listdir(RAW_DIR):


    # person_dir — путь к папке с оригинальными фото. 
    # out_person_dir — путь к папке, куда сохранить обрезанные фото. 
    person_dir = os.path.join(RAW_DIR, person)
    out_person_dir = os.path.join(OUT_DIR, person)
    os.makedirs(out_person_dir, exist_ok = True)

    # Дополнитеьный цикл в котром проходим по изображениям в папке
    for img_name in tqdm(os.listdir(person_dir),desc=person):

        # img_path — путь к исходному фото.   
        # out_path — путь, куда сохранить вырезанное лицо.        
        img_path = os.path.join(person_dir,img_name)
        out_path = os.path.join(out_person_dir,img_name)

        try:
            # открываем изображение и приводим к RGB
            img = Image.open(img_path).convert('RGB')
            face = mtcnn(img)                  # пропускаем иозбражение через mtcnn

            if face is not None:               # eсли лицо найдено — переводим тензор в PIL.Image и сохраняем.
                to_pil_image(face).save(out_path)
        
        except Exception as e: 
             print(f"Ошибка при обработке {img_path}: {e}")




