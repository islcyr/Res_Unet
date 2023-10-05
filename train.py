# -*- coding: utf-8 -*-
"""
@Time ： 2023/4/25 2:30
@Auth ： Yin yanquan
@File ：train.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader

from model import WSLnet
from data_loader import dataset


def train():
    lr = 0.001
    epochs = 100
    batch_size = 16
    steps = [150000, 40000000]

    dataset_train = dataset('../dataset/SAN_5w/tam',
                            '../dataset/SAN_5w/data_384.txt',
                            '../dataset/SAN_5w/mask')
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0,
                              pin_memory=False)

    print("CUDA available: ", torch.cuda.is_available())
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = WSLnet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=0.1, last_epoch=-1)

    model_save_path = './results'
    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    log_save_path = './log'
    if not os.path.isdir(log_save_path):
        os.makedirs(log_save_path)
    fid = open('./log/%s.txt' % (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))), 'wt')

    model.train()

    idx = 0
    t0 = time.time()
    for epoch in range(0, epochs + 1):
        for batch_idx, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device).float()

            optimizer.zero_grad()
            output = model(data)
            output = output.squeeze(1)
            output = torch.sigmoid(output)

            criterion = nn.BCELoss()
            # print(output.shape, label.shape)
            loss = criterion(output.view(-1), label.view(-1)).to(device)

            loss.backward()
            optimizer.step()
            scheduler.step()

            if batch_idx % 100 == 0:
                lr = optimizer.param_groups[0]['lr']

                print('Epoch[{}/{}]\tlr:{:f}\tloss:{:.6f}\ttime:{:.1f}'
                      .format(epoch, epochs, lr, loss.item(), time.time() - t0)
                      )
                fid.write('Epoch[{}/{}]\tlr:{:f}\tloss:{:.6f}\ttime:{:.1f}\n'
                          .format(epoch, epochs, lr, loss.item(), time.time() - t0)
                          )
                fid.flush()
                t0 = time.time()

            idx += 1
            if idx % 1000 == 0:
                save_path = '%s/model-%02d.pkl' % (model_save_path, idx)
                torch.save(model.state_dict(), save_path)
                print(save_path)

    fid.close()


if __name__ == '__main__':
    torch.cuda.empty_cache()
    train()
