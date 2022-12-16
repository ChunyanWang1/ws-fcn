import torch
import torch.nn as nn
import torch.nn.functional as F

def weight_init(module):
    for n, m in module.named_children():
        #print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()

class SF2(nn.Module):
    def __init__(self):
        super(SF2, self).__init__()

        self.conv1h = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn1h   = nn.BatchNorm2d(256)
        self.conv2h = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn2h   = nn.BatchNorm2d(256)
        self.conv3h = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3h   = nn.BatchNorm2d(256)
        self.conv4h = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4h   = nn.BatchNorm2d(256)

        self.conv1v = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn1v   = nn.BatchNorm2d(256)
        self.conv2v = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn2v   = nn.BatchNorm2d(256)
        self.conv3v = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3v   = nn.BatchNorm2d(256)
        self.conv4v = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4v   = nn.BatchNorm2d(256)
        self.initialize()

    def forward(self, left, down):
        if down.size()[2:] != left.size()[2:]:
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear')
        out1h = F.relu(self.bn1h(self.conv1h(left )), inplace=True)
        out2h = F.relu(self.bn2h(self.conv2h(out1h)), inplace=True)
        out1v = F.relu(self.bn1v(self.conv1v(down )), inplace=True)
        out2v = F.relu(self.bn2v(self.conv2v(out1v)), inplace=True)
        fuse  = out2h*out2v
        out3h = F.relu(self.bn3h(self.conv3h(fuse )), inplace=True)+out1h
        out4h = F.relu(self.bn4h(self.conv4h(out3h)), inplace=True)
        out3v = F.relu(self.bn3v(self.conv3v(fuse )), inplace=True)+out1v
        out4v = F.relu(self.bn4v(self.conv4v(out3v)), inplace=True)
        return out4h, out4v

    def initialize(self):
        weight_init(self)






class AlignModule(nn.Module):
    def __init__(self, inplane, outplane):
        super(AlignModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 2, kernel_size=3, padding=1, bias=False)


    def forward(self, x):
        low_feature, h_feature= x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature= self.down_h(h_feature)
        h_feature = F.interpolate(h_feature,size=size, mode="bilinear", align_corners=False)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm   # 扭曲坐标

        output = F.grid_sample(input, grid,align_corners=True)   # 双线性插值得到高分辨率特征 ,align_corners=True
        return output
