import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
from resnet import ResNet18, ResNet34
from resnet_imagenet import resnet50


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class proj_head(nn.Module):
    def __init__(self, ch, use_norm=False, output_cnt=None):
        super(proj_head, self).__init__()
        self.in_features = ch

        if output_cnt is None:
            output_cnt = ch

        self.fc1 = nn.Linear(ch, ch)
        self.bn1 = nn.BatchNorm1d(ch)

        if use_norm:
            self.fc2 = NormedLinear(ch, output_cnt)
        else:
            self.fc2 = nn.Linear(ch, output_cnt, bias=False)
        self.bn2 = nn.BatchNorm1d(output_cnt)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)

        return x


class SimCLR(nn.Module):
    def __init__(self, num_class=100, network='resnet18'):
        super(SimCLR, self).__init__()
        self.backbone = self.get_backbone(network)(num_class, use_norm=False)
        self.backbone.in_planes = self.backbone.fc.weight.shape[1]
        self.backbone.fc = nn.Identity()

        self.projector = proj_head(self.backbone.in_planes, False, 128)
  
        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )

    @staticmethod
    def get_backbone(backbone_name):
        return {'resnet18': ResNet18,
                'resnet34': ResNet34,}[backbone_name]

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        return {'z1': z1, 'z2': z2}

class SimCLR_imagenet(nn.Module):
    def __init__(self, num_class=100):
        super(SimCLR_imagenet, self).__init__()
        self.backbone = resnet50(pretrained=False, imagenet=True, num_classes=num_class)
        self.backbone.in_planes = self.backbone.fc.weight.shape[1]
        self.backbone.fc = nn.Identity()

        self.projector = proj_head(self.backbone.in_planes, False, 128)

        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        return {'z1': z1, 'z2': z2}