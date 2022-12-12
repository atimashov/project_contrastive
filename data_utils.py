from PIL import Image
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageNet
import random
import torch
from torch.utils.data import Subset, DataLoader, Dataset
from torchvision.transforms.transforms import RandomApply
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import json

def compute_train_transform(seed=123456):
    """
    This function returns a composition of data augmentations to a single training image.
    1. Random crop of 224 x 224
    """
    random.seed(seed)
    torch.random.manual_seed(seed)
    
    # Transformation that applies color jitter with brightness=0.4, contrast=0.4, saturation=0.4, and hue=0.1
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1) # NOTE: not sure if I need it
    
    train_transform = transforms.Compose([
        # Step 1: Randomly crop 224x224.
        transforms.RandomCrop(224),
        # Step 2: Horizontally flip the image with probability 0.5
        transforms.RandomHorizontalFlip(0.5), 
        # Step 3: With a probability of 0.8, apply color jitter (you can use "color_jitter" defined above.
        # transforms.RandomApply([color_jitter], p = 0.8),
        # Step 4: With a probability of 0.2, convert the image to grayscale
        # transforms.RandomGrayscale(0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    return train_transform


def compute_detect_transform(img, boxes):
    """
    This function returns a composition of data augmentations to a single training image.
    1. Random crop of 224 x 224
    """
    # PIL to torch tensor
    base_transform = transforms.ToTensor()
    final_transform = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    img = base_transform(img)

    # random crop
    _, H, W = img.shape
    if boxes.shape[0] == 0:
        w, h = random.randint(0, W - 224), random.randint(0, H - 224)
        img = img[:, h: (h + 224), w: (w + 224)]
        img = final_transform(img)
        return img, boxes


    # NOTE: I want to be sure that I will get at least one box
    bb_idx = random.choice(range(boxes.shape[0]))
    x0_start, x0_end = int(boxes[bb_idx, 0] - boxes[bb_idx, 2] / 2), int(boxes[bb_idx, 0] + boxes[bb_idx, 2] / 2)
    y0_start, y0_end = int(boxes[bb_idx, 1] - boxes[bb_idx, 3] / 2), int(boxes[bb_idx, 1] + boxes[bb_idx, 3] / 2)
    # random crop height:
    if max(y0_end - 224, 0) + 1 <= min(y0_start, H - 224):
        h = random.randint(max(y0_end - 224, 0), min(y0_start, H - 224))
    elif min(y0_start, H - 224) - 1 >= 0:
        h = min(y0_start, H - 224) - 1
    else:
        h = max(y0_end - 224, 0)
    # random crop width
    if max(x0_end - 224, 0) + 1 <= min(x0_start, W - 224):
        w = random.randint(max(x0_end - 224, 0), min(x0_start, W - 224))
    elif min(x0_start, W - 224) - 1 >= 0:
        w = min(x0_start, W - 224) - 1
    else:
        w = max(x0_end - 224, 0)
    img = img[:,h: (h + 224), w: (w + 224)]
    # keep only not empty boxes
    x_start, x_end = boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 0] + boxes[:, 2] / 2
    y_start, y_end = boxes[:, 1] - boxes[:, 3] / 2, boxes[:, 1] + boxes[:, 3] / 2
    cond_x_start = torch.logical_and(w <= x_start, x_start < w + 224)
    cond_x_end = torch.logical_and(w <= x_end, x_end < w + 224)
    cond_x = torch.logical_or(cond_x_start, cond_x_end)
    cond_y_start = torch.logical_and(h <= y_start, y_start < h + 224)
    cond_y_end = torch.logical_and(h <= y_end, y_end < h + 224)
    cond_y = torch.logical_or(cond_y_start, cond_y_end)
    cond = torch.logical_and(cond_x, cond_y)
    # shift x & y
    x_start, x_end = x_start[cond] - w, x_end[cond] - w
    x_start, x_end = torch.where(x_start >= 0, x_start, 0), torch.where(x_end <= 223, x_end, 223) # NOTE: can it be a problem here?
    y_start, y_end = y_start[cond] - h, y_end[cond] - h
    y_start, y_end = torch.where(y_start >= 0, y_start, 0), torch.where(y_end <= 223, y_end, 223) # NOTE: can it be a problem here?
    boxes = torch.stack([(x_start + x_end) / 2, (y_start + y_end) / 2, x_end - x_start, y_end - y_start], -1)
    # normalize
    boxes = boxes / 224

    # modify w and h

    img = final_transform(img)
    # TODO: should I make boxes [0, 1]
    return img, boxes


def compute_test_transform():
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    return test_transform


class BarrierReefPair(Dataset):
    """
        Great Barrier Reef Dataset.
        video_0: 0-12,347
        video_1: 0-11,374
        video_2: 0-10,759
    """
    # TODO: add labels
    def __init__(self, root, train = True, transform = None):
        self.root = root
        self.train = train
        self.transform = transform

        # data frame containing labels for the images
        df = pd.read_csv('{}/train.csv'.format(self.root))

        # create images & labels
        # - I will use 90% of the data in train
        # - I will use every 15-th frame in train avoiding duplicated data. (idx % 15 = 0)
        self.imgs = []
        self.labels = []
        for i in range(3):
            path = '{}/video_{}'.format(self.root, i)
            n = len(os.listdir(path))
            if train:
                self.imgs.extend(
                    [os.path.join(path, img) for i, img in enumerate(os.listdir(path)) if i <= 0.9 * n]
                )
            else:
                self.imgs.extend(
                    [os.path.join(path, img) for i, img in enumerate(os.listdir(path)) if i > 0.9 * n]
                )
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # load images and targets
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")

        x_i = None
        x_j = None

        if self.transform is not None:
            x_i = self.transform(img)
            x_j = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return x_i, x_j, # target

class BarrierReefDetect(torch.utils.data.Dataset):
    def __init__(
            self, root, train = True, transform = None, S = 7, B = 2
    ):
        self.root = root
        self.train = train
        self.transform = transform

        self.imgs = []
        self.labels = []
        for i in range(3):
            path = '{}/video_{}'.format(self.root, i)
            n = len(os.listdir(path))
            if train:
                self.imgs.extend(
                    [os.path.join(path, img) for i, img in enumerate(os.listdir(path)) if i <= 0.9 * n and '.jpg' in img]
                )
            else:
                self.imgs.extend(
                    [os.path.join(path, img) for i, img in enumerate(os.listdir(path)) if i > 0.9 * n and '.jpg' in img]
                )

        # data frame containing labels for the images
        self.df = pd.read_csv('{}/train.csv'.format(self.root))
        self.S, self.B = S, B

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        image = Image.open(img_path)  # in PIL

        helper_id = img_path.find('video_') + 6
        img_id = '{}-{}'.format(img_path[helper_id], img_path[(helper_id + 2):-4])
        boxes_string = self.df[self.df['image_id'] == img_id]['annotations'].values[0]
        boxes = [[box['x'], box['y'], box['width'], box['height']] for box in json.loads(boxes_string.replace("'", '"'))]
        if len(boxes) > 0: boxes = torch.tensor(boxes)
        else: boxes = torch.zeros(0, 4)

        if self.transform:
            image, boxes = self.transform(image, boxes) # TODO: make normal augmentation for object detection

        # convert to cells
        label_matrix = torch.zeros((self.S, self.S, self.B * 5))
        for box in torch.unbind(boxes):
            x, y, width, height = box.tolist()

            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = width * self. S, height * self.S

            if label_matrix[i, j, 0] == 0:
                label_matrix[i, j, 0] = 1
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, 1:5] = box_coordinates
        return image, label_matrix

class CIFAR10Pair(CIFAR10):
    """
    CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        x_i = None
        x_j = None

        if self.transform is not None:
            x_i = self.transform(img)
            x_j = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return x_i, x_j, target

if __name__ == '__main__':
    if False:
        train_transform = compute_train_transform()
        train_data = BarrierReefPair(root='tensorflow-great-barrier-reef', train=True, transform=train_transform)
        # train_data = torch.utils.data.Subset(train_data, list(np.arange(int(len(train_data) * percentage))))
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=True,
                                  drop_last=True)

        train_bar = tqdm(train_loader)
        for data_pair in train_bar:
            x_i, x_j = data_pair
            print(x_i.shape, x_j.shape)
            break
    else:
        train_transform = compute_detect_transform
        train_data = BarrierReefDetect(root='great-barrier-reef-small', train=False, transform=train_transform)
        # train_data = torch.utils.data.Subset(train_data, list(np.arange(int(len(train_data) * percentage))))
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1, pin_memory=True,
                                  drop_last=True)

        # train_bar = tqdm(train_loader)
        i, j = 0, 0
        print('LENGTH: ', train_data.__len__())
        # dataset = iter(train_loader)
        # while i < 100:
        train_bar = tqdm(train_loader)
        for data in train_bar:
            # j += 1
            # print('ITER: ', j, i)
            img, boxes = data
            # if boxes.shape[0] > 0:
            #     i += 1
        # for data_pair in train_bar:
        #     img, boxes = data_pair
        #     print(img.shape, boxes.shape)
        #     break

