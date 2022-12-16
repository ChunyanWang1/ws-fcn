import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    """refine global information"""

    def __init__(self, in_channels=4096,out_channels=256,ks=7):
        super(GCN, self).__init__()

        self.NormLayer = nn.BatchNorm2d
        self.from_scratch_layers = []


        self._init_params(in_channels,out_channels,ks)


    def _init_params(self,in_channels,out_channels,ks):

        self.conv1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(ks,1), padding=(ks//2,0))
        self.conv1_2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1,ks), padding=(0, ks // 2))
        #self.bn = nn.BatchNorm2d(out_channels)
        self.conv2_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(ks,1),  padding=(ks//2,0))
        self.conv2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1,ks), padding=(0, ks // 2))
        #self.relu = nn.ReLU()



        torch.nn.init.kaiming_normal_(self.conv1_1.weight)
        torch.nn.init.kaiming_normal_(self.conv1_2.weight)
        torch.nn.init.kaiming_normal_(self.conv2_1.weight)
        torch.nn.init.kaiming_normal_(self.conv2_2.weight)
        self.from_scratch_layers = [self.conv1_1,self.conv1_2,self.conv2_1,self.conv2_2]


    def forward(self, x):
        """Forward pass

        Args:
            x: features
        """
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x2 = self.conv2_1(x)
        x2 = self.conv2_2(x2)
        #x2=self.bn(x2)

        res = x2 + x1

        return res


class BoundaryRefine(nn.Module):
    def __init__(self,dim):
        super(BoundaryRefine, self).__init__()
        self.relu=nn.ReLU(inplace=True)
        self.conv1=nn.Conv2d(dim,dim,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(dim,dim,kernel_size=3,padding=1)

    def forward(self, x):
        r1=self.conv1(x)
        r1=self.relu(r1)
        r1=self.conv2(r1)
        out=x+r1
        return out
