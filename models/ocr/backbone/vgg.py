import torch
from einops import rearrange
from torch import nn
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter


class VGG(nn.Module):
    def __init__(self, ss, ks, hidden, pretrained=True, dropout=0.5):
        super(VGG, self).__init__()
        cnn = models.vgg19_bn(pretrained=pretrained)
        pool_idx = 0
        for i, layer in enumerate(cnn.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                cnn.features[i] = torch.nn.AvgPool2d(
                    kernel_size=ks[pool_idx], stride=ss[pool_idx], padding=0
                )
                pool_idx += 1

        self.features = cnn.features
        self.dropout = nn.Dropout(dropout)
        self.last_conv_1x1 = nn.Conv2d(512, hidden, 1)

    def forward(self, x):
        """
        Shape:
            - x: (N, C, H, W)
            - output: (W, N, C)
        """

        conv = self.features(x)
        conv = self.dropout(conv)
        conv = self.last_conv_1x1(conv)

        # conv = rearrange(conv, 'b d h w -> b d (w h)')
        conv = conv.transpose(-1, -2)
        conv = conv.flatten(2)
        conv = conv.permute(-1, 0, 1)
        return conv


def vgg19_bn(ss, ks, hidden, pretrained=True, dropout=0.5):
    ss = [tuple(i) for i in ss]
    ks = [tuple(i) for i in ks]
    return VGG(ss, ks, hidden, pretrained, dropout)
