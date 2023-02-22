import torch
from loguru import logger
from torch import nn

from . import vgg


class CNN(nn.Module):
    def __init__(self, cfg):
        super(CNN, self).__init__()
        self.cfg = cfg
        if self.cfg.model.backbone == "vgg19_bn":
            self.model = vgg.vgg19_bn(
                cfg.model.cnn.ss,
                cfg.model.cnn.ks,
                cfg.model.cnn.hidden,
                cfg.model.cnn.pretrained,
            )
        else:
            logger.error("Backbone have not supported yet!")

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for name, param in self.model.features.named_parameters():
            if name != "last_conv_1x1":
                param.requires_grad = False

    def unfreeze(self):
        for param in self.model.features.parameters():
            param.requires_grad = True
