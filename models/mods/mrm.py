import torch
import torch.nn as nn
import torch.nn.functional as F
class MRM(nn.Module):
    def __init__(self, class_num=20, kernel=3, H=81, W=81,bn_momentum=0.1,track_running_stats=True):
        super(MRM, self).__init__()
        self.class_num=class_num
        self.kernel=kernel
        self.H=H
        self.W=W
        self.iter_num=10
        self.feature_num = self.kernel * self.kernel * self.class_num
        self.inter_feature=256
        self.norm_sum = 1
        self.offset = kernel // 2
        #self.toplayer = nn.Conv2d(1024,256, kernel_size=1, stride=1)

        self.layer1 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256, momentum=bn_momentum, track_running_stats=track_running_stats),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256, momentum=bn_momentum, track_running_stats=track_running_stats),
        )



        # self.smooth3 = nn.Sequential(
        #     nn.Conv2d(intermediate_feature, intermediate_feature, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(intermediate_feature, momentum=bn_momentum, track_running_stats=track_running_stats),
        #
        # )
        #
        # # out
        self.smooth1 = nn.Sequential(
            nn.Conv2d(256, self.feature_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.feature_num, momentum=bn_momentum, track_running_stats=track_running_stats),
        )
        self.smooth2 = nn.Sequential(
            nn.Conv2d(256, self.feature_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.feature_num, momentum=bn_momentum, track_running_stats=track_running_stats),
        )
        self.smooth3 = nn.Sequential(
            nn.Conv2d(256, self.feature_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.feature_num, momentum=bn_momentum, track_running_stats=track_running_stats),
        )

        self.activate = nn.ReLU(inplace=True)

    def feature_com(self,fea):
        B,C,H,W=fea.size()
        weight = fea.permute(0, 2, 3, 1)
        weight = weight.view(B, H, W,self.class_num, self.kernel, self.kernel)

        weight = self._norm_weight(weight)

        weight = weight.view(B, H, W, -1)
        weight = weight.permute(0, 3, 1, 2).view(B, weight.shape[3], -1)

        return weight

    def feature_fusion(self,fea):
        fea3 =  self.activate(self.smooth1(fea[0]))
        _, _, H, W = fea[0].size()

        fea2= F.interpolate(fea[1], size=(H, W), mode='bilinear', align_corners=True)
        fea1 = F.interpolate(fea[2], size=(H, W), mode='bilinear', align_corners=True)
        fea2= self.activate(self.smooth2(self.layer2(fea2)))
        fea1= self.activate(self.smooth3(self.layer1(fea1)))

        weight3 = self.feature_com(fea3)


        weight2 = self.feature_com(fea2)

        weight1 = self.feature_com(fea1)

        weight=(weight3+weight2+weight1)/3

        return weight



    def _norm_weight(self, weight):
        weight_size = weight.size()
        weight_kernel = weight.view(weight.size(0), weight.size(1),
                                    weight.size(2), weight.size(3), -1)
        weight_kernel_sum = weight_kernel.sum(4, keepdim=True)
        mask = weight_kernel_sum[:, :, :, :] == 0
        weight_kernel_sum[mask] = 1

        norm_weight = weight_kernel / weight_kernel_sum * self.norm_sum
        norm_weight = norm_weight.view(weight_size)

        return norm_weight




    def forward(self,x,fea):
        weight=self.feature_fusion(fea)
        #x = F.unfold(x, kernel_size=self.kernel, padding=self.offset)
        for iter in range(self.iter_num):
            x = F.unfold(x, kernel_size=self.kernel, padding=self.offset)
            temp = torch.mul(x,weight)
            temp = temp.view(temp.shape[0], self.class_num, self.kernel, self.kernel, self.H, self.W)
            temp = temp.permute(0, 1, 4, 5, 2, 3).view(temp.shape[0], self.class_num, self.H, self.W, -1)
            x = temp.sum(4)
        return x


