import torch
import torch.nn as nn
import numpy as np
from models.grl_t import DomainAdaptationModule

class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=False, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_c, out_c,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class multikernel_dilated_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.c1 = Conv2D(in_c, out_c, kernel_size=1, padding=0)
        self.c2 = Conv2D(in_c, out_c, kernel_size=3, padding=1)
        self.c3 = Conv2D(in_c, out_c, kernel_size=7, padding=3)
        self.c4 = Conv2D(in_c, out_c, kernel_size=11, padding=5)
        self.s1 = Conv2D(out_c*4, out_c, kernel_size=1, padding=0)

        self.d1 = Conv2D(out_c, out_c, kernel_size=3, padding=1, dilation=1)
        self.d2 = Conv2D(out_c, out_c, kernel_size=3, padding=3, dilation=3)
        self.d3 = Conv2D(out_c, out_c, kernel_size=3, padding=7, dilation=7)
        self.d4 = Conv2D(out_c, out_c, kernel_size=3, padding=11, dilation=11)
        self.s2 = Conv2D(out_c*4, out_c, kernel_size=1, padding=0, act=False)
        #self.s3 = Conv2D(in_c, out_c, kernel_size=1, padding=0, act=False)

        self.ca = ChannelAttention(out_c)
        self.sa = SpatialAttention()

    def forward(self, x):
        x0 = x
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        x = torch.cat([x1, x2, x3, x4], axis=1)
        s = self.s1(x)


        x1 = self.d1(s)
        x2 = self.d2(s)
        x3 = self.d3(s)
        x4 = self.d4(s)
        x = torch.cat([x1, x2, x3, x4], axis=1)
        x = self.s2(x)
        #s = self.s3(x0)

        x = self.relu(x+s)
        x = x * self.ca(x)
        x = x * self.sa(x)

        return x
    
class M_U_Net(nn.Module):
    """
    The Block Module in paper, a modified U-Net
    """
    def __init__(self, in_channel=9, out_channel=3):
        """
        :param in_channel:
        :param out_channel:
        """
        super(M_U_Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 90, 3, 1, 1),
            nn.BatchNorm2d(90),
            nn.ReLU(),
            nn.Conv2d(90, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            *(
                (
                    nn.Conv2d(64, 64, 3, 1, 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU()
                )*2
            )
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            *(
                (
                    nn.Conv2d(128, 128, 3, 1, 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU()
                ) * 4
            ),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.conv4 = nn.Sequential(
            *(
                (
                    nn.Conv2d(64, 64, 3, 1, 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU()
                )*2
            ),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_channel, 3, 1, 1)
        )
        # multikernel_dilated_conv
        self.c1 = Conv2D(16, 64)
        self.c2 = Conv2D(32, 64)
        self.c3 = Conv2D(64, 64)
        self.c4 = Conv2D(128, 64)
        self.m1 = multikernel_dilated_conv(64, 32)
        self.m2 = multikernel_dilated_conv(64, 32)
        self.m3 = multikernel_dilated_conv(64, 64)
        self.m4 = multikernel_dilated_conv(64, 32)


    def forward(self, data, ref):
        """
        :param data: noisy frames
        :param ref: reference frame that is the middle frame of noisy frames
        :return:
        """
        conv1 = self.conv1(data)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)


        c2 = self.c2(conv1)
        c3 = self.c3(conv2)
        # c3 = self.c3(conv3)

        #s1 = self.m1(c2)  #32   60
        s2 = self.m2(c2) #64      120
        s3 = self.m3(c3)  #32      240
        # s4 = self.m4(c4)  #16


        conv4 = self.conv4(conv3 + s3)
        conv5 = self.conv5(conv4 + s2)
        
        return conv5+ref

class M_U_Net_GRL(nn.Module):
    """
    The Block Module in paper, a modified U-Net
    """
    def __init__(self, in_channel=9, out_channel=3):
        """
        :param in_channel:
        :param out_channel:
        """
        super(M_U_Net_GRL, self).__init__()

        self.DomainAdaptationModule = DomainAdaptationModule()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 90, 3, 1, 1),#没有c_in输入时，inchannals=9
            nn.BatchNorm2d(90),
            nn.ReLU(),
            nn.Conv2d(90, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            *(
                (
                    nn.Conv2d(64, 64, 3, 1, 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU()
                )*2
            )
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            *(
                (
                    nn.Conv2d(128, 128, 3, 1, 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU()
                ) * 4
            ),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.conv4 = nn.Sequential(
            *(
                (
                    nn.Conv2d(64, 64, 3, 1, 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU()
                )*2
            ),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_channel, 3, 1, 1)
        )

        # multikernel_dilated_conv
        self.c1 = Conv2D(16, 64)
        self.c2 = Conv2D(32, 64)
        self.c3 = Conv2D(64, 64)
        self.c4 = Conv2D(128, 64)
        self.m1 = multikernel_dilated_conv(64, 32)
        self.m2 = multikernel_dilated_conv(64, 32)
        self.m3 = multikernel_dilated_conv(64, 64)
        self.m4 = multikernel_dilated_conv(64, 32)

    def forward(self, data, ref):
        """
        :param data: noisy frames
        :param ref: reference frame that is the middle frame of noisy frames
        :return:
        """
        conv1 = self.conv1(data)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        
        domain = self.DomainAdaptationModule(conv2)

        c2 = self.c2(conv1)
        c3 = self.c3(conv2)
        # c3 = self.c3(conv3)

        #s1 = self.m1(c2)  #32   60
        s2 = self.m2(c2) #64      120
        s3 = self.m3(c3)  #32      240
        # s4 = self.m4(c4)  #16


        conv4 = self.conv4(conv3 + s3)
        conv5 = self.conv5(conv4 + s2)

        return conv5+ref, domain

