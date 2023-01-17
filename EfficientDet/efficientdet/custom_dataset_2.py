import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2
import pathlib
import json
import random

class CustomDataset(Dataset):

    def __init__(self, transform=None, is_test=False, keep_difficult=False,
                 label_file=None):
        if is_test:
            image_sets_file = "gdrive/MyDrive/capstone/efficientdet/EfficientDet/data/val.txt"
            self.test_num = 1
        else:
            image_sets_file = "gdrive/MyDrive/capstone/efficientdet/EfficientDet/data/train.txt"
            self.test_num = 0

        self.transform = transform

        self.ids = CustomDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        # if the labels file exists, read in the class names
        label_file_name = "gdrive/MyDrive/capstone/efficientdet/EfficientDet/data/crack-model-labels.txt"

        if os.path.isfile(label_file_name):
            classes = list()

            file = open(label_file_name, mode="r")
            for line in file.readlines():
                line = line.strip()
                classes.append(line)
            # prepend BACKGROUND as first class
            classes.insert(0, 'BACKGROUND')
            classes = [elem.replace(" ", "") for elem in classes]

            self.class_names = tuple(classes)
            file.close()

        else:
            self.class_names = (
            'BACKGROUND', 'ConcreteCrack', 'Exposure', 'Spalling')# , 'PaintDamage', 'Efflorescene', 'SteelDefect'

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        annotation, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            annotation = annotation[is_difficult == 0]
        image = self._read_image(image_id)
        H,W,_ = image.shape

        x1,y1,x2,y2 = annotation[0][:4]

        num = 0
        while num == 4:
            i = random.randint(20,100)

            if num == 0 :
                if x1 - i < 0:
                    x_min = x1
                else:
                    x_min = x1-i
            elif num == 1:
                if y1 - i < 0:
                    y_min = y1
                else:
                    y_min = y1 - i
            elif num == 2:
                if x2 + i > W:
                    x_max = x2
                else:
                    x_max = x2+i
            elif num == 3:
                if y2 + i > H:
                    y_max = y2
                else:
                    y_max = y2+i
            num+=1

        image = image[y_min:y_max,x_min,x_max,:]


        sample = {'img': image, 'annot': annotation}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):

        real_name = image_id.split('.')[0]
        json_name = real_name + '.json'

        is_difficult = []
        annotation = []

        with open(f"./data/json/{json_name}", 'rt', encoding='UTF8') as f:

            data = json.load(f)
            W = data['images'][0]['width']
            H = data['images'][0]['height']


            while True:

                i = random.randint(0,len(data['annotations']))
                class_name = data['annotations'][i]['attributes']['class']

                if class_name not in self.class_dict:
                    continue

                x1 = float(data['annotations'][i]['bbox'][0] / W) * 512
                y1 = float(data['annotations'][i]['bbox'][1] / H) * 512
                x2 = float((data['annotations'][i]['bbox'][0] + data['annotations'][i]['bbox'][2]) / W) * 512
                y2 = float((data['annotations'][i]['bbox'][1] + data['annotations'][i]['bbox'][3]) / H) * 512

                if x1 > x2 or y1 > y2:
                    print(f"{image_id} error 발생")
                    raise ValueError

                if x1 == 0 or x2 == 0 or y1 == 0 or y2 == 0:
                    if x1 == 0:
                        x1 == 3
                    if x2 == 0:
                        x2 == 3
                    if y1 == 0:
                        y1 == 3
                    if y2 == 0:
                        y1 == 3

                if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
                    print(f"{image_id} error 발생")
                    raise ValueError

                annotation.append([x1, y1, x2, y2, self.class_dict[class_name]])
                is_difficult_str = 0
                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

                break

        return (np.array(annotation, dtype=np.float32),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id):

        if self.test_num == 1:
            directory = "total_512"
        else:
            directory = "total_512"

        image = cv2.imread("/content/total_512/{}/{}".format(directory, image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image.astype(np.float32) / 255.


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        # H, W 순서 때문에 이렇게 새로 array 만들고 넣는 것 같은데... 이렇게 해야 하나?
        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object): ## 좌우 반전
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x: # 0 ~ 1 사이의 random인 값에서 0.5 보다 작으면 실행 즉, 50% 로 실행
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :] # -1 이라서 거꾸로 만들어지는 것

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp # x축으로 뒤집어지는 box 보정

            sample = {'img': image, 'annot': annots}

        return sample

