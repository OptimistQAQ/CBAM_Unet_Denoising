import numpy as np
import torch
from torch import nn
from torch.nn import init

"""

    卷积块注意块 convolutional Block Attention Module(CBAM)
    一种简单而有效的前馈卷积神经网络注意模块
    旨在通过同时使用通道和空间关注机制来提高模型的表示能力
    沿着两个独立的维度、通道和空间顺序推断注意力图，然后将注意力图乘以输入特征图以进行自适应特征细化

"""
class ChannelAttention(nn.Module):
    """
    该模块计算通道关注。它在空间维度上应用最大池化和平均池化,
    并使用两个卷积层来学习通道关注权重。对这些权重应用了Sigmoid函数
    """
    def __init__(self,channel,reduction=16):
        super().__init__()
        # 自适应最大池化和自适应平均池化，用于在空间维度上进行池化操作，将张量的高度和宽度维度降为1
        # 将输入张量的高度和宽度维度降为1。这是为了计算通道注意力的一部分，以便在通道维度上学习权重，而不受空间维度的影响
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        # 使用两个卷积层来学习通道注意力权重 MLP层
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        # 通过Sigmoid函数来处理这些权重,将学到的通道注意力权重归一化到0到1之间
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :

        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        # 传递给se模块，以计算通道注意力权重
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)

        output=self.sigmoid(max_out+avg_out)
        return output

class SpatialAttention(nn.Module):
    """
    关键思想是将最大值和平均值作为特征，通过卷积操作学习哪些空间位置对于任务的关键性
    """
    def __init__(self,kernel_size=7):
        super().__init__()
        # 保持输出特征图与输入特征图的相同空间尺寸
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        # 输入张量 x 首先沿通道维度使用 torch.max 函数获取最大值和通过 torch.mean 函数获取平均值
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        # 将这两个值连接在一起，创建一个具有两个通道的新特征图
        result=torch.cat([max_result,avg_result],1)
        # 通过卷积层 conv 处理连接的特征图，并通过Sigmoid激活函数将结果归一化到0到1之间，并返回作为输出
        output=self.conv(result)

        output=self.sigmoid(output)
        return output


# 该模块计算空间关注。它获取沿通道维度的最大值和平均值，将它们连接在一起，然后通过卷积层和Sigmoid激活函数
class CBAMBlock(nn.Module):

    def __init__(self, channel=512,reduction=16,kernel_size=49):
        """
        channel、reduction 和 kernel_size,
        分别表示输入特征张量的通道数、通道注意力中的通道减少因子和空间注意力中的卷积核大小
        """
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)

    # 用于初始化CBAMBlock中卷积层的权重。它使用不同的初始化方法来初始化权重。
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    def forward(self, x):
        # 其中 b 是批量大小，c 是通道数，_ 表示高度和宽度
        b, c, _, _ = x.size()
        # print(b)  # 50
        # print(c)  # 512
        residual=x
        # 通过通道注意力 ca 处理输入 x，得到通道注意力加权的特征
        out=x*self.ca(x)
        # 通过空间注意力 sa 处理加权的特征，得到空间注意力加权的特征
        out=out*self.sa(out)
        # 将空间注意力加权的特征与残差连接，返回最终的输出
        return out+residual


# if __name__ == '__main__':
#     # 创建了一个形状为 (50, 512, 7, 7) 的示例输入张量，表示包含50张图像、每张图像有512个通道，空间分辨率为7x7
#     input=torch.randn(50,512,7,7)
#     kernel_size=input.shape[2]
#     cbam = CBAMBlock(channel=512,reduction=16,kernel_size=kernel_size)
#     output=cbam(input)
#     print(output.shape)

    