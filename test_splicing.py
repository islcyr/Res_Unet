# -*- coding: utf-8 -*-
"""
@Time ： 2023/4/25 3:04
@Auth ： Yin yanquan
@File ：test_splicing.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from model import WSLnet

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def test_one():
    model_path = r'./results/model-315000.pkl'
    for i in range(1,101):
        print(i)
        img_fn = '../dataset/SAN_5w/tam/%s.jpg' % i
        img = Image.open(img_fn)
        imgs = np.array(img)
        imgs = imgs.astype(np.float32) / 255.
        imgs = imgs.transpose([2, 0, 1])
        data = np.expand_dims(imgs, axis=0)
        data = torch.from_numpy(data)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = WSLnet().to(device)
        pretrain = torch.load(model_path)
        model.load_state_dict(pretrain, strict=True)
        model.eval()

        predict_ = model(data.to(device))
        predict_ = torch.sigmoid(predict_)

        predict = predict_.cpu().detach().numpy()
        predict_image = np.transpose(predict[0], (1, 2, 0)) * 255
        ret, predict_image = cv2.threshold(predict_image, 100, 255, cv2.THRESH_BINARY)

        cv2.imwrite('./image/test_%s.png' % i,predict_image.astype(int))


if __name__ == '__main__':
    torch.cuda.empty_cache()
    test_one()