import torch
import torch.nn as nn
import torch.nn.functional as F


class RBM(nn.Module):
    """refine feature module"""

    def __init__(self, in_channels,out_channels):
        super(RBM, self).__init__()

        self.NormLayer = nn.BatchNorm2d
        self.from_scratch_layers = []


        self._init_params(in_channels,out_channels)


    def _init_params(self,in_channels,out_channels):

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        #self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.sigmod = nn.Sigmoid()


        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        self.from_scratch_layers = [self.conv1,self.conv2]


    def forward(self, x1,x2):
        """Forward pass

        Args:
            x: features
        """

        #x1, x2 = x  # high, low

        x = torch.cat([x1, x2], dim=1)
        x = self.global_pooling(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmod(x)
        x2 = x * x2
        #res = x2 + x1
        return x2+x1