import torch
import torch.nn as nn
import torch.nn.functional as F


class RFM(nn.Module):
    """refine feature module"""

    def __init__(self, in_channels,out_channels):
        super(RFM, self).__init__()

        self.NormLayer = nn.BatchNorm2d
        self.from_scratch_layers = []


        self._init_params(in_channels,out_channels)


    def _init_params(self,in_channels,out_channels):

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        if not self.bn.weight is None:
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        self.from_scratch_layers = [self.conv1,self.conv2,self.conv3,self.bn]


    def forward(self, x):
        """Forward pass

        Args:
            x: features
        """

        x = self.conv1(x)
        res = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        return self.relu(x + res)