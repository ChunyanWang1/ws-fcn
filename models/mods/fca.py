import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#from dropblock import DropBlock2D
                   
class GCN_Module(nn.Module):
    """refine global information"""

    def __init__(self, in_channels=4096,out_channels=256,ks=7,dilation=1):
        super(GCN_Module, self).__init__()

        self.NormLayer = nn.BatchNorm2d
        self.from_scratch_layers = []
        if dilation==1:
            dilation1=0
        elif dilation==12:
            dilation1=dilation
        elif dilation==24:
            dilation1=2*dilation
        else:
            dilation1=3*dilation

        self.conv1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(ks, 1),padding=(dilation1, 0),dilation=dilation,bias=False)
        self.conv1_2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, ks), padding=(0, dilation1),dilation=dilation,bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv2_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,ks), padding=(0,dilation1),dilation=dilation,bias=False)
        self.conv2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=(ks,1), padding=(dilation1,0),dilation=dilation,bias=False)
        self.relu = nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv1_1.weight)
        torch.nn.init.kaiming_normal_(self.conv1_2.weight)
        torch.nn.init.kaiming_normal_(self.conv2_1.weight)
        torch.nn.init.kaiming_normal_(self.conv2_2.weight)




    def forward(self, x):
        """Forward pass

        Args:
            x: features
        """
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x2 = self.conv2_1(x)
        x2 = self.conv2_2(x2)
        res = x2 + x1
        return res



class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

class FCA(nn.Module):
    def __init__(self, inplanes, output_stride, BatchNorm):
        super(FCA, self).__init__()

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError


        
        self.sa=SpatialAttention(3)
        self.ca=ChannelAttention(256)

        self.fca1 = GCN_Module(inplanes,256,ks=1,dilation=1)
        self.fca2 = GCN_Module(inplanes,256,ks=3,dilation=12)
        self.fca3 = GCN_Module(inplanes,256,ks=5,dilation=24)
        self.fca4 = GCN_Module(inplanes,256,ks=7,dilation=36)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                              nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                              BatchNorm(256),
                                              nn.ReLU())



        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
 
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
     
        self._init_weight()





    def forward(self, x):

        x1 = self.fca1(x)
        x1=self.ca(x1)*x1

        x2 = self.fca2(x)
        x2=self.ca(x2)*x2

        x3 = self.fca3(x)
        x3=self.ca(x3)*x3

        x4 = self.fca4(x)
        x4=self.ca(x4)*x4

        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        
        
        x = self.conv1(x)
        x = self.bn1(x)
        x=self.sa(x)*x+x
        
        return self.dropout(x)


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if not m.weight is None:
                    m.weight.data.fill_(1)
                else:
                    print("FCA has not weight: ", m)

                if not m.bias is None:
                    m.bias.data.zero_()
                else:
                    print("FCA has not bias: ", m)


def build_fca(backbone, output_stride, BatchNorm):
    return FCA(backbone, output_stride, BatchNorm)



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7),"kernel size must be 3 or 7"
        padding=3 if kernel_size==7 else 1

        self.conv=nn.Conv2d(2,1,kernel_size,padding=padding,bias=False)
        self.sigmoid=nn.Sigmoid()
        self._init_weight()

    def forward(self, x):
        avgout=torch.mean(x,dim=1,keepdim=True)
        maxout,_=torch.max(x,dim=1,keepdim=True)
        x=torch.cat([avgout,maxout],dim=1)
        x=self.conv(x)
        return self.sigmoid(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)


class ChannelAttention(nn.Module):
    def __init__(self, inplanes,ratio=1):
        super(ChannelAttention, self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.max_pool=nn.AdaptiveMaxPool2d(1)
        self.sharedMLP=nn.Sequential(
                                     nn.Conv2d(inplanes,inplanes//ratio,1,bias=False),nn.ReLU(),
                                     nn.Conv2d(inplanes//ratio,inplanes//ratio,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
        self._init_weight()

    def forward(self, x):
        avgout=self.sharedMLP(self.avg_pool(x))
        maxout=self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout+maxout)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
