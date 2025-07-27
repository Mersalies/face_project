import os
import random

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset



class TripletFaceDataset(Dataset):

    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
            ])
        else:
            self.transform = transform

        #сбор списка папок по людям
        self.people = os.listdir(root_dir)
        self.people = [p for p in self.people if os.path.isdir(os.path.join(root_dir, p))]

        self.image_paths = { # Всегда используйте image_paths
                    person: [
                              os.path.join(root_dir,person,img)
                              for img in os.listdir(os.path.join(root_dir,person))
                            ]
                    for person in self.people
                  }


         # Убираем людей с <2 фото (т.к. нужны минимум anchor+positive)
        self.image_paths = {k: v for k, v in self.image_paths.items() if len(v) >= 2}
        self.people = list(self.image_paths.keys())

    def __len__(self):
        return 100000
    

    def __getitem__(self, index):
        
        # выбираем человека А(anchor/positive)
        person = random.choice(self.people)
        imgs = random.sample(self.image_paths[person],2)
        anchor_path, positive_path = imgs[0], imgs[1]


        # Выбираем человека B ≠ A (negative)
        negative_person = random.choice([p for p in self.people if p != person])
        negative_path = random.choice(self.image_paths[negative_person])


        # Загружаем изображения
        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')


        # Применяем преобразования (resize, normalization и т.п.)
        if self.transform :
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor,positive,negative
