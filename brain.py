import torch
from glob import glob
from PIL import Image
import random


class BrainTrain(torch.utils.data.Dataset):
    def __init__(self, transform, id=1):
        self.transform = transform
        if id == 1:
            self.image_paths = glob('./Br35H/dataset/train/normal/*')
        else:
            self.image_paths = glob('./brats/dataset/train/normal/*')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img


class BrainTest(torch.utils.data.Dataset):
    def __init__(self, transform, test_id=1):

        self.transform = transform
        self.test_id = test_id

        test_normal_path = glob('./Br35H/dataset/test/normal/*')
        test_anomaly_path = glob('./Br35H/dataset/test/anomaly/*')

        self.test_path = test_normal_path + test_anomaly_path
        self.test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)

        if self.test_id == 2:
            test_normal_path = glob('./brats/dataset/test/normal/*')
            test_anomaly_path = glob('./brats/dataset/test/anomaly/*')

            self.test_path = test_normal_path + test_anomaly_path
            self.test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)

        combined = list(zip(self.test_path, self.test_label))
        random.seed(0)
        random.shuffle(combined)
        self.test_path, self.test_label = zip(*combined)
        self.test_path = list(self.test_path)
        self.test_label = list(self.test_label)

    def __len__(self):
        return len(self.test_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.test_path[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, self.test_label[idx], img_path
