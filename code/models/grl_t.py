import torch
import torch.nn.functional as F
from torch import nn

class DAImgHead(nn.Module):
    """
    Adds a simple Image-level Domain Classifier head
    """

    def __init__(self):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            USE_FPN (boolean): whether FPN feature extractor is used
        """
        super(DAImgHead, self).__init__()

        self.conv1_da = nn.Conv2d(64, 32, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv2d(32, 1, kernel_size=1, stride=1)

        for l in [self.conv1_da, self.conv2_da]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        t = F.relu(self.conv1_da(x))
        return self.conv2_da(t)

class _GradientScalarLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):    #input:1×512×1×1
        ctx.weight = weight
        return input.view_as(input)            #正向传播没变化

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return ctx.weight*grad_input, None     #反向传播时梯度乘以*-0.1

gradient_scalar = _GradientScalarLayer.apply


class GradientScalarLayer(torch.nn.Module):
    def __init__(self, weight):
        super(GradientScalarLayer, self).__init__()
        self.weight = weight

    def forward(self, input):
        return gradient_scalar(input, self.weight)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "weight=" + str(self.weight)
        tmpstr += ")"
        return tmpstr
    

class DomainAdaptationModule(torch.nn.Module):
    """
    Module for Domain Adaptation Component. Takes feature maps from the backbone and instance
    feature vectors, domain labels and proposals. Works for both FPN and non-FPN.
    """

    def __init__(self):
        super(DomainAdaptationModule, self).__init__()

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.sigmoid_gap = nn.Sigmoid()

        self.grl_img = GradientScalarLayer(-1.0)    #正向传播是没变化，反向传播时梯度乘以-0.1
        self.imghead = DAImgHead()
        
    def forward(self, img_features):
        """
        Arguments:
            img_features (list[Tensor]): features computed from the images that are
                used for computing the predictions.
            da_ins_feature (Tensor): instance-level feature vectors
            da_ins_labels (Tensor): domain labels for instance-level feature vectors
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """

        img_grl_fea = self.grl_img(self.gap(img_features))  # 经过grl的image-level feature    #1×512×1×1
        # b=self.imghead(img_grl_fea)
        da_img_features = self.sigmoid_gap(self.imghead(img_grl_fea))  # 对image-level feature的每个层都压成1*1   #1×1×1×1

        return da_img_features