import torch
import torch.nn.functional as F
import torch.nn as nn

from functools import partial


#
# Helper modules
#
class LocalAffinity(nn.Module):

    def __init__(self, dilations=[1]):
        super(LocalAffinity, self).__init__()
        self.dilations = dilations
        weight = self._init_aff()
        self.register_buffer('kernel', weight)

    def _init_aff(self):
        # initialising the shift kernel
        weight = torch.zeros(8, 1, 3, 3)

        for i in range(weight.size(0)):
            weight[i, 0, 1, 1] = 1

        weight[0, 0, 0, 0] = -1
        weight[1, 0, 0, 1] = -1
        weight[2, 0, 0, 2] = -1

        weight[3, 0, 1, 0] = -1
        weight[4, 0, 1, 2] = -1

        weight[5, 0, 2, 0] = -1
        weight[6, 0, 2, 1] = -1
        weight[7, 0, 2, 2] = -1

        self.weight_check = weight.clone()

        return weight

    def forward(self, x):

        self.weight_check = self.weight_check.type_as(x)
        assert torch.all(self.weight_check.eq(self.kernel))

        B, K, H, W = x.size()
        x = x.view(B * K, 1, H, W)


        x_aff=F.conv2d(x, self.kernel,stride=1,padding=1)
        # x_pad = F.pad(x, [1]*4, mode='replicate')
        # print('x_pad', x_pad.size())
        # x_aff = F.conv2d(x, self.kernel,dilation=0)
        #print('x_aff',x_aff.size())
        # x_affs.append(x_aff)
        return x_aff.view(B, K, -1, H, W)
        # for d in self.dilations:
        #     x_pad = F.pad(x, [d]*4, mode='replicate')
        #     x_aff = F.conv2d(x_pad, self.kernel, dilation=d)
        #     x_affs.append(x_aff)
        #
        #
        # x_aff = torch.cat(x_affs, 1)
        # return x_aff.view(B,K,-1,H,W)
        # for d in self.dilations:
        #     x_pad = F.pad(x, [d] * 4, mode='replicate')
        #     x_aff = F.conv2d(x_pad, self.kernel, dilation=d)
        #     y_aff = x_aff.view(B, K, -1, H, W)
        #     y_aff=(y_aff*y_aff[:,:, 4, :, :].unsqueeze(2))#.sum(1,keepdim=True)
        #     x_aff = y_aff[:,:,torch.arange(y_aff.size(2)) != 4,:,:]
        #
        #     x_affs.append(x_aff)
        #
        # x_aff = torch.cat(x_affs, 2)
        # return x_aff #x_aff.view(B, K, -1, H, W)


class LocalAffinityCopy(LocalAffinity):

    def _init_aff(self):
        # initialising the shift kernel
        weight = torch.zeros(8, 1, 3, 3)
        weight[0, 0, 0, 0] = 1
        weight[1, 0, 0, 1] = 1
        weight[2, 0, 0, 2] = 1

        weight[3, 0, 1, 0] = 1
        weight[4, 0, 1, 2] = 1

        weight[5, 0, 2, 0] = 1
        weight[6, 0, 2, 1] = 1
        weight[7, 0, 2, 2] = 1

        self.weight_check = weight.clone()
        return weight

    # def forward(self, x):
    #     self.weight_check = self.weight_check.type_as(x)
    #     assert torch.all(self.weight_check.eq(self.kernel))
    #
    #     B, K, H, W = x.size()
    #     x = x.view(B * K, 1, H, W)
    #
    #     x_affs = []
    #     for d in self.dilations:
    #         x_pad = F.pad(x, [d] * 4, mode='replicate')
    #         x_aff = F.conv2d(x_pad, self.kernel, dilation=d)
    #         y_aff = x_aff.view(B, K, -1, H, W)
    #         x_aff = y_aff[:, :, torch.arange(y_aff.size(2)) != 4, :, :]
    #         x_affs.append(x_aff)
    #
    #     x_aff = torch.cat(x_affs, 2)
    #     #return x_aff.view(B, K, -1, H, W)
    #     return x_aff


class LocalStDev(LocalAffinity):

    def _init_aff(self):
        weight = torch.zeros(9, 1, 3, 3)
        weight.zero_()

        weight[0, 0, 0, 0] = 1
        weight[1, 0, 0, 1] = 1
        weight[2, 0, 0, 2] = 1

        weight[3, 0, 1, 0] = 1
        weight[4, 0, 1, 1] = 1
        weight[5, 0, 1, 2] = 1

        weight[6, 0, 2, 0] = 1
        weight[7, 0, 2, 1] = 1
        weight[8, 0, 2, 2] = 1

        self.weight_check = weight.clone()
        return weight

    def forward(self, x):
        # returns (B,K,P,H,W), where P is the number
        # of locations
        #x = super(LocalStDev, self).forward(x)
        self.weight_check = self.weight_check.type_as(x)
        assert torch.all(self.weight_check.eq(self.kernel))

        B, K, H, W = x.size()
        x = x.view(B * K, 1, H, W)

        x_affs = []
        for d in self.dilations:
            x_pad = F.pad(x, [d] * 4, mode='replicate')
            x_aff = F.conv2d(x_pad, self.kernel, dilation=d)
            #assert torch.equal(x,x_aff[:, 4, :, :].unsqueeze(1)), "dimension mismatch"
            y_aff = x_aff.view(B,K,-1,H,W)

            y_aff = torch.norm(y_aff, dim=1, keepdim=True) * torch.norm(
                y_aff[:, :, 4, :, :], dim=1, keepdim=True).unsqueeze(2)
            y_aff=y_aff[:,:,torch.arange(y_aff.size(2))!=4,:,:]
            x_affs.append(y_aff)

        x_aff = torch.cat(x_affs, 2)
        x= x_aff.view(B, 1, -1, H, W)
        #x=x_aff.view(B,)

        #return x.std(2, keepdim=True)
        return x


class LocalAffinityAbs(LocalAffinity):

    def forward(self, x):
        x = super(LocalAffinityAbs, self).forward(x)
        return torch.abs(x)
        #return x


#
# PAMR module
#
class Affinity(nn.Module):

    def __init__(self, num_iter=1, dilations=[1]):
        super(Affinity, self).__init__()

        self.num_iter = num_iter
        self.aff_x = LocalAffinityAbs(dilations)
        self.aff_m = LocalAffinityCopy(dilations)
        #self.aff_std = LocalStDev(dilations)

    def forward(self, x, mask):
        mask = F.interpolate(mask, size=x.size()[-2:], mode="bilinear", align_corners=True)

        # x: [BxKxHxW]
        # mask: [BxCxHxW]
        B, K, H, W = x.size()
        _, C, _, _ = mask.size()

        #x_std = self.aff_std(x)

        x = -self.aff_x(x)
        #x = -self.aff_x(x).sum(1,keepdim=True) / (1e-8 +  x_std)
        #print(x.size())
        #x=x.sum()
        x = x.mean(1, keepdim=True)

        x = F.softmax(x, 2)
        _, _, m, _, _ = x.size()

        # alpha=0.01
        # x[x<alpha]=0

        for _ in range(self.num_iter):
            m = self.aff_m(mask)  # [BxCxPxHxW]
            mask = (m * x).sum(2)

        # xvals: [BxCxHxW]
        return mask
