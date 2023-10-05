# -*- coding: utf-8 -*-
"""
@Time ： 2023/4/25 2:04
@Auth ： Yin yanquan
@File ：data_loader.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import json


# np.set_printoptions(threshold=np.inf)

class dataset(Dataset):
    def __init__(self, root_folder, txt_path, root_folder2=None):
        self.img_list = []
        self.label_list = []
        self.root_folder = root_folder + '/'
        if root_folder2 is None:
            self.root_folder2 = self.root_folder
        else:
            self.root_folder2 = root_folder2 + '/'
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                s = line.split(',')
                self.img_list.append(s[0])
                self.label_list.append(s[1].replace('\n', ''))
        if len(self.img_list) == 0:
            print('warning: find none images')
        else:
            print('datasize', len(self.label_list))
        self.kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

    # read images
    def cv_imread(self, filePath, color=cv2.IMREAD_COLOR):
        image = np.fromfile(filePath, dtype=np.uint8)
        cv_img = cv2.imdecode(image, color)
        return cv_img

    def __getitem__(self, index):
        # image process
        img = self.cv_imread(self.root_folder + self.img_list[index])
        label = self.cv_imread(self.root_folder2 + self.label_list[index], cv2.IMREAD_GRAYSCALE)

        img = img.astype(np.float32) / 255.
        img = img.transpose([2, 0, 1])
        ret, label = cv2.threshold(label, 100, 255, cv2.THRESH_BINARY)
        label = label // 255

        return img, label

    def __len__(self):
        return len(self.label_list)


if __name__ == '__main__':

    epochs = 1
    # data = dataset('./dataset/show', './dataset/show/data384.txt')
    dataset_train = dataset('./dataset/SAN_8w/tam',
                            './dataset/SAN_8w/data_384.txt',
                            './dataset/SAN_8w/mask')

    loader = DataLoader(dataset_train, batch_size=4, shuffle=True, drop_last=True, num_workers=0, pin_memory=False)
    # loader = DataLoader(data, batch_size=4, shuffle=True, drop_last=True, num_workers=0, pin_memory=False)

    for ep in range(epochs):
        for batch_idx, (data, data_edge, label, label_edge) in enumerate(loader):
            img = data.detach().numpy()
            data_edge = data_edge.detach().numpy()
            mask = label.detach().numpy()
            label_edge = label_edge.detach().numpy()

            img = img[0] * 255
            img = img.transpose([1, 2, 0])

            data_edge = data_edge[0] * 255
            data_edge = data_edge.astype(int)
            data_edge[data_edge > 0] += 20
            data_edge[data_edge > 255] = 255
            data_edge = data_edge.astype(np.uint8)

            mask = mask[0] * 255
            mask = mask.astype(int)
            mask[mask > 0] += 20
            mask[mask > 255] = 255
            mask = mask.astype(np.uint8)

            # cv2.imwrite('./dataset/show/label_edge_1.png', label_edge.astype(np.uint8))
            label_edge = label_edge[0] * 255
            label_edge = label_edge.astype(int)
            label_edge[label_edge > 0] += 20
            label_edge[label_edge > 255] = 255
            label_edge = label_edge.astype(np.uint8)

            print('write')
            print(img.shape)
            # cv2.imwrite('./dataset/show/data.png', img.astype(np.uint8))
            # cv2.imwrite('./dataset/show/data_edge.png', data_edge.astype(np.uint8))
            # cv2.imwrite('./dataset/show/label.png', mask.astype(np.uint8))
            # cv2.imwrite('./dataset/show/label_edge.png', label_edge.astype(np.uint8))
            break
