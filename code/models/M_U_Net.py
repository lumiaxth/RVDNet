import torch
import torch.nn as nn
import numpy as np
from models.grl_t import DomainAdaptationModule

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

    def forward(self, data, ref):
        """
        :param data: noisy frames
        :param ref: reference frame that is the middle frame of noisy frames
        :return:
        """
        conv1 = self.conv1(data)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3 + conv2)
        conv5 = self.conv5(conv4 + conv1)
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

    def forward(self, data, ref):
        """
        :param data: noisy frames
        :param ref: reference frame that is the middle frame of noisy frames
        :return:
        """
        conv1 = self.conv1(data)
        conv2 = self.conv2(conv1)
        domain = self.DomainAdaptationModule(conv2)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3 + conv2)
        conv5 = self.conv5(conv4 + conv1)
        return conv5+ref, domain

