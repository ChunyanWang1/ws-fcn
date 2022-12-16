# -*- coding: utf-8 -*-
# @Time    : 2020-2-20 18:49:41
# @Author  : Lart Pang
# @FileName: DCM.py
# @Project : PyTorchCoding
# @GitHub  : https://github.com/lartpang

import torch
import torch.nn as nn
import torch.nn.functional as F


class DCM(nn.Module):
    def __init__(self, in_C, out_C,dilations):
        super(DCM, self).__init__()
        self.ks = 3#[1, 3, 5]
        self.mid_C = in_C//4
        self.dilations=dilations
        self.num=len(dilations)
        self.ger_kernel=nn.Sequential(nn.AdaptiveAvgPool2d(self.ks), nn.Conv2d(in_C, self.mid_C, kernel_size=1))
        self.trans_=nn.Conv2d(in_C, self.mid_C, kernel_size=1)
        self.fuse_inside=nn.Conv2d(self.mid_C, self.mid_C, 1)
        self.fuse_outside =nn.Conv2d(self.num*self.mid_C + in_C, out_C, 1)


    def forward(self, x, y):
        """
        x: 被卷积的特征
        y: 用来生成卷积核
        """
        feats_branches = [x]
        kernel = self.ger_kernel(y)
        kernel_single = kernel.split(1, dim=0)
        x_inside = self.trans_(x)
        x_inside_single = x_inside.split(1, dim=0)


        for d in self.dilations:
            feat_single = []
            for kernel_single_item, x_inside_single_item \
                    in zip(kernel_single, x_inside_single):
                feat_inside_single = self.fuse_inside(
                    F.conv2d(
                        x_inside_single_item,
                        weight=kernel_single_item.transpose(0, 1),
                        bias=None,
                        stride=1,
                        padding=d,#self.ks // 2,
                        dilation=d,
                        groups=self.mid_C
                    )
                )
                feat_single.append(feat_inside_single)
            feat_single = torch.cat(feat_single, dim=0)

            feats_branches.append(feat_single)

        feats_branches=torch.cat(feats_branches, dim=1)


        return self.fuse_outside(feats_branches)



# if __name__ == '__main__':
#     x = torch.randn(4, 2048, 20, 20)
#     y = torch.randn(4, 2048, 20, 20)
#     dcm = DCM(in_C=2048, out_C=256,dilations=[1,12,24,36])
#     print(dcm(x, y).size())
