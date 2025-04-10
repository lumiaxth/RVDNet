import torch
import torch.nn.functional as F
import torch.nn as nn


class _UpProjection(nn.Sequential):

    def __init__(self, num_input_features, num_output_features, use_bn=False):
        super(_UpProjection, self).__init__()

        self.conv1 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(num_output_features)

        self.conv2 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_output_features)

        self.use_bn = use_bn

    def forward(self, x, size):
        x = F.upsample(x, size=size, mode='bilinear', align_corners=True)

        if not self.use_bn:
            x_conv1 = self.relu(self.conv1(x))
            bran1 = self.conv1_2(x_conv1)
            bran2 = self.conv2(x)
        else:
            x_conv1 = self.relu(self.bn1(self.conv1(x)))
            bran1 = self.bn1_2(self.conv1_2(x_conv1))
            bran2 = self.bn2(self.conv2(x))

        out = self.relu(bran1 + bran2)

        return out


class E_resnet(nn.Module):

    def __init__(self, original_model, num_features=2048, use_bn=False):
        super(E_resnet, self).__init__()
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool

        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

        self.use_bn = use_bn

    def forward(self, x):
        x = self.conv1(x)

        if self.use_bn:
            x = self.bn1(x)

        x = self.relu(x)
        x_block0 = x
        x = self.maxpool(x)

        x_block1 = self.layer1(x)
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)

        return x_block0, x_block1, x_block2, x_block3, x_block4


class D_resnet(nn.Module):

    def __init__(self, num_features=2048, use_bn=False):
        super(D_resnet, self).__init__()
        # self.conv = nn.Conv2d(num_features, num_features //
        #                       2, kernel_size=1, stride=1, bias=False)
        # num_features = num_features // 2

        self.conv = nn.Conv2d(num_features, num_features, kernel_size=1, stride=1, bias=False)
        # num_features = num_features

        self.bn = nn.BatchNorm2d(num_features)

        self.up1 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2, use_bn=use_bn)
        num_features = num_features // 2

        self.up2 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2, use_bn=use_bn)
        num_features = num_features // 2

        self.up3 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2, use_bn=use_bn)
        num_features = num_features // 2

        self.up4 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features, use_bn=use_bn)

        self.up5 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 4)

        self.use_bn = use_bn

    def forward(self, x_block0, x_block1, x_block2, x_block3, x_block4):
        if self.use_bn:
            x_d0 = F.relu(self.bn(self.conv(x_block4))) + x_block4
        else:
            x_d0 = F.relu(self.conv(x_block4)) + x_block4

        x_d1 = self.up1(x_d0, [x_block3.size(2), x_block3.size(3)]) + x_block3
        x_d2 = self.up2(x_d1, [x_block2.size(2), x_block2.size(3)]) + x_block2
        x_d3 = self.up3(x_d2, [x_block1.size(2), x_block1.size(3)]) + x_block1
        x_d4 = self.up4(x_d3, [x_block0.size(2), x_block0.size(3)]) + x_block0
        x_d5 = self.up5(x_d4, [x_block0.size(2) * 2, x_block0.size(3) * 2])

        return x_d5


class MFF(nn.Module):

    def __init__(self, block_channel, num_features=64):
        super(MFF, self).__init__()

        self.up1 = _UpProjection(
            num_input_features=block_channel[0], num_output_features=16)

        self.up2 = _UpProjection(
            num_input_features=block_channel[1], num_output_features=16)

        self.up3 = _UpProjection(
            num_input_features=block_channel[2], num_output_features=16)

        self.up4 = _UpProjection(
            num_input_features=block_channel[3], num_output_features=16)

        self.conv = nn.Conv2d(
            num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x_block1, x_block2, x_block3, x_block4, size, use_bn=False):
        x_m1 = self.up1(x_block1, size)
        x_m2 = self.up2(x_block2, size)
        x_m3 = self.up3(x_block3, size)
        x_m4 = self.up4(x_block4, size)

        if use_bn:
            x = self.bn(self.conv(torch.cat((x_m1, x_m2, x_m3, x_m4), 1)))
        else:
            x = self.conv(torch.cat((x_m1, x_m2, x_m3, x_m4), 1))
        x = F.relu(x)

        return x
