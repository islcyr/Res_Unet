# -*- coding: utf-8 -*-
"""
@Time ： 2023/4/24 20:05
@Auth ： Yin yanquan
@File ：model.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class WSLnet(nn.Module):
    def __init__(self):
        super(WSLnet, self).__init__()
        bn = 32  # base output number

        # ----------------------- Encoder ----------------------- #
        ## --------------- UNet3+ --------------
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, bn, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv1_maxpooling = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=4, stride=2, padding=1),  # 卷积代替池化
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv2_maxpooling = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=4, stride=2, padding=1),  # 卷积代替池化
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv3_maxpooling = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=4, stride=2, padding=1),  # 卷积代替池化
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv4_maxpooling = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=4, stride=2, padding=1),  # 卷积代替池化
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )

        # ----------------------- Middle ----------------------- #
        ## --------------- UNet3+ --------------
        self.conv5 = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )

        # ----------------------- Decoder ----------------------- #
        ## --------------- UNet3+ --------------
        self.se4 = SELayer(channel=bn, reduction=16)

        '''stage 4d'''
        # h1->384*384, hd4->48*48, Pooling 8 times
        self.h1_PT_hd4 = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=8, stride=8, padding=0),
            nn.Conv2d(bn, bn, 3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )

        # h2->192*192, hd4->48*48, Pooling 4 times
        self.h2_PT_hd4 = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=4, stride=4, padding=0),
            nn.Conv2d(bn, bn, 3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )

        # h3->96*96, hd4->48*48, Pooling 2 times
        self.h3_PT_hd4 = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(bn, bn, 3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )

        # h4->48*48, hd4->48*48, Concatenation
        self.h4_Cat_hd4 = nn.Sequential(
            nn.Conv2d(bn, bn, 3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )

        # hd5->24*24, hd4->48*48, Upsample 2 times
        self.hd5_UT_hd4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(bn, bn, 3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.fusion4d = nn.Sequential(
            nn.Conv2d(5 * bn, 5 * bn, 3, padding=1),
            nn.BatchNorm2d(5 * bn),
            nn.ReLU(True)
        )

        self.se3 = SELayer(channel=5 * bn, reduction=16)

        '''stage 3d'''
        # h1->384*384, hd3->96*96, Pooling 4 times
        self.h1_PT_hd3 = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=4, stride=4, padding=0),
            nn.Conv2d(bn, bn, 3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )

        # h2->192*192, hd3->96*96, Pooling 2 times
        self.h2_PT_hd3 = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(bn, bn, 3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )

        # h3->96*96, hd3->96*96, Concatenation
        self.h3_Cat_hd3 = nn.Sequential(
            nn.Conv2d(bn, bn, 3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )

        # hd4->48*48, hd4->96*96, Upsample 2 times
        self.hd4_UT_hd3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(5 * bn, bn, 3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )

        # hd5->24*24, hd4->96*96, Upsample 4 times
        self.hd5_UT_hd3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(bn, bn, 3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.fusion3d = nn.Sequential(
            nn.Conv2d(5 * bn, 5 * bn, 3, padding=1),  # 16
            nn.BatchNorm2d(5 * bn),
            nn.ReLU(True)
        )

        self.se2 = SELayer(channel=5 * bn, reduction=16)

        '''stage 2d '''
        # h1->384*384, hd2->192*192, Pooling 2 times
        self.h1_PT_hd2 = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(bn, bn, 3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )

        # h2->192*192, hd2->192*192, Concatenation
        self.h2_Cat_hd2 = nn.Sequential(
            nn.Conv2d(bn, bn, 3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )

        # hd3->96*96, hd2->192*192, Upsample 2 times
        self.hd3_UT_hd2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(5 * bn, bn, 3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )

        # hd4->48*48, hd2->192*192, Upsample 4 times
        self.hd4_UT_hd2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(5 * bn, bn, 3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )

        # hd5->24*24, hd2->192*192, Upsample 8 times
        self.hd5_UT_hd2 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear'),
            nn.Conv2d(bn, bn, 3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.fusion2d = nn.Sequential(
            nn.Conv2d(5 * bn, 5 * bn, 3, padding=1),
            nn.BatchNorm2d(5 * bn),
            nn.ReLU(True)
        )

        self.se1 = SELayer(channel=5 * bn, reduction=16)

        '''stage 1d'''
        # h1->384*384, hd1->384*384, Concatenation
        self.h1_Cat_hd1 = nn.Sequential(
            nn.Conv2d(bn, bn, 3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )

        # hd2->192*192, hd1->384*384, Upsample 2 times
        self.hd2_UT_hd1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(5 * bn, bn, 3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )

        # hd3->96*96, hd1->384*384, Upsample 4 times
        self.hd3_UT_hd1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(5 * bn, bn, 3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )

        # hd4->48*48, hd1->384*384, Upsample 8 times
        self.hd4_UT_hd1 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear'),
            nn.Conv2d(5 * bn, bn, 3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )

        # hd5->24*24, hd1->384*384, Upsample 16 times
        self.hd5_UT_hd1 = nn.Sequential(
            nn.Upsample(scale_factor=16, mode='bilinear'),
            nn.Conv2d(bn, bn, 3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )

        # 未结合前的结果用于计算残差块
        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.fusion1d = nn.Sequential(
            nn.Conv2d(5 * bn, 5 * bn, 3, padding=1),
            nn.BatchNorm2d(5 * bn),
            nn.ReLU(True)
        )

        # ----------------------- Output ----------------------- #
        ## --------------- UNet3+ --------------
        self.conv_end_img_edge = nn.Conv2d(5 * bn, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, data):
        U1 = self.conv1(data)
        U1_mp = self.conv1_maxpooling(U1)

        U2 = self.conv2(U1_mp)
        U2_mp = self.conv2_maxpooling(U2)

        U3 = self.conv3(U2_mp)
        U3_mp = self.conv3_maxpooling(U3)

        U4 = self.conv4(U3_mp)
        U4_mp = self.conv4_maxpooling(U4)

        U5 = self.conv5(U4_mp)

        U5_se = self.se4(U5)

        h1_PT_hd4 = self.h1_PT_hd4(U1)
        h2_PT_hd4 = self.h2_PT_hd4(U2)
        h3_PT_hd4 = self.h3_PT_hd4(U3)
        h4_Cat_hd4 = self.h4_Cat_hd4(U4)
        hd5_UT_hd4 = self.hd5_UT_hd4(U5_se)
        hd4 = torch.cat([h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4], 1)
        hd4_out = self.fusion4d(hd4)

        hd4_out_se = self.se3(hd4_out)

        h1_PT_hd3 = self.h1_PT_hd3(U1)
        h2_PT_hd3 = self.h2_PT_hd3(U2)
        h3_Cat_hd3 = self.h3_Cat_hd3(U3)
        hd4_UT_hd3 = self.hd4_UT_hd3(hd4_out_se)
        hd5_UT_hd3 = self.hd5_UT_hd3(U5_se)
        hd3 = torch.cat([h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3], 1)
        hd3_out = self.fusion3d(hd3)

        hd3_out_se = self.se2(hd3_out)

        h1_PT_hd2 = self.h1_PT_hd2(U1)
        h2_Cat_hd2 = self.h2_Cat_hd2(U2)
        hd3_UT_hd2 = self.hd3_UT_hd2(hd3_out_se)
        hd4_UT_hd2 = self.hd4_UT_hd2(hd4_out_se)
        hd5_UT_hd2 = self.hd5_UT_hd2(U5_se)
        hd2 = torch.cat([h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2], 1)
        hd2_out = self.fusion2d(hd2)

        hd2_out_se = self.se1(hd2_out)

        h1_Cat_hd1 = self.h1_Cat_hd1(U1)
        hd2_UT_hd1 = self.hd2_UT_hd1(hd2_out_se)
        hd3_UT_hd1 = self.hd3_UT_hd1(hd3_out_se)
        hd4_UT_hd1 = self.hd4_UT_hd1(hd4_out_se)
        hd5_UT_hd1 = self.hd5_UT_hd1(U5_se)
        hd1 = torch.cat([h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1], 1)
        hd1_out = self.fusion1d(hd1)

        end_img_edge = self.conv_end_img_edge(hd1_out)
        return end_img_edge


if __name__ == '__main__':
    net = WSLnet()
    data = np.ones([1, 3, 256, 384], np.float32)
    data = torch.from_numpy(data)
    r = net(data)
    print(r)