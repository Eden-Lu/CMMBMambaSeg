
from __future__ import annotations
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba

# from mmcv.ops import DeformConv3d
#
# class DeformableConv3D(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3):
#         super().__init__()
#         # 计算填充量，确保输入和输出大小一致
#         padding = (kernel_size - 1) // 2
#
#         # 偏移量生成卷积
#         self.offset_conv = nn.Conv3d(
#             in_channels,
#             3 * kernel_size ** 3,  # 偏移量通道数
#             kernel_size=kernel_size,
#             stride=1,
#             padding=padding
#         )
#
#         # 可形变卷积
#         self.deform_conv = DeformConv3d(
#             in_channels,
#             out_channels,
#             kernel_size=kernel_size,
#             stride=1,
#             padding=padding
#         )
#
#     def forward(self, x):
#         # 生成偏移量
#         offset = self.offset_conv(x)
#         # 使用偏移量进行可形变卷积
#         out = self.deform_conv(x, offset)
#         return out



def ini_weights(module_list:list):
    for m in module_list:
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')       # 恺明初始化策略初始化卷积核
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)                        # 卷积核偏置初始化为
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 1)                          # 归一化层权重初始化为1
            nn.init.constant_(m.bias, 0)                            # 归一化层权重初始化为0
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)                    # 标准正太分布初始化全连接层
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)                        # 全连接层偏置初始化为0



class LayerNorm(nn.Module):

    # channels_last（默认）：输入张量形状为 (batch_size, height, width, channels)。
    # channels_first：输入张量形状为 (batch_size, channels, height, width)。

    # normalized_shape是需要进行归一化维度的具体大小，即channel维度的大小
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError                                       # 报错提醒
        # 将normalized_shape转换成元组格式，因为F.layer_norm要求normalized_shape为元组格式
        self.normalized_shape = (normalized_shape, )

        # ini_weights(self.module_list())

    # 这个循环保证，无论输入那种类型，都是对channel这一维度进行归一化
    def forward(self, x):
        if self.data_format == "channels_last":
            # 这个函数默认对最后一维，即channel维进行归一化
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x


class RMB(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            # bimamba_type="v3",
            # nslices=num_slices,
        )

    def forward(self, x):
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out = out + x_skip

        return out

class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, ):
        super().__init__()
        # 卷积核大小为1，只改变通道数，将输入张量的通道数从hidden_size->mlp_dim
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class HGDC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):
        x_residual = x

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)

        return x + x_residual


class DCNM1(nn.Module):
    # depths：深度，决定每层的mamba数量   out_indices:控制那个阶段输出特征，这里表示每个阶段都要输出特征
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        # nn.ModuleList是PyTorch内置的一种特殊的容器，用于存储多个nn.Module子模块。
        # 这里是创建了一个列表downsample_layers,用来储存下采样层
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
        )
        # 将模型起始部分的stem层，添加到下采样层（downsample_layer）当中
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            # 将stem后的三个下采样层储存在列表downsample_layers
            self.downsample_layers.append(downsample_layer)

            # 编码器四层，为什么储存三层，猜测原因：stem和前三层要用于解码器残差连接

        # 类似上面，创建列表，分别储存各阶段的不同操作
        self.stages = nn.ModuleList()  # 主要是mamba操作
        self.gscs = nn.ModuleList()  # 主要是卷积操作
        num_slices_list = [64, 32, 16, 8]  # 指定每个阶段的分片数，用来控制MambaLayer的分块数量
        cur = 0  # 用来追踪当前阶段的特征层数目
        for i in range(4):  # i=[0,1,2,3]
            gsc = HGDC(dims[i])  # GSC只有一个参数:in_channel

            stage = nn.Sequential(
                # manba定义了两个参数：dim和num_slices,别的参数都固定了
                # *是用来重复操作，重复次数由内部for循环控制，这里depths=2，即mamba重复两次
                *[RMB(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
                # 这里需要修改，[64,64,64,]->[256,256,20]
            )

            # 上述创建的列表里面是空的，经过for循环后，可以将每层的操作储存在相应列表中
            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]  # 累计记录已经创建的特征提取层数

        self.out_indices = out_indices  # 用来指定输出阶段，便于后续模块处理

        # 同样，创建一个列表储存mlp操作
        self.mlps = nn.ModuleList()
        for i_layer in range(4):  # i_layer=[0,1,2,3]
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            # MlpChannel定义两个参数：in_channel,out_channel
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))
    def forward_features(self, x1):
        outs1 = []
        for i in range(4):  # i=[0,1,2,3]
            x1 = self.downsample_layers[i](x1)
            x1 = self.gscs[i](x1)
            x1 = self.stages[i](x1)

            # out_indices作用：列表表示要在这些位置进行归一化和mlp操作
            if i in self.out_indices:  # 判断i是否在列表out_indices=[0,1,2,3]中
                # getattr用于动态获取归一化层norm[i],通过名字就可以获取该层归一化操作
                norm_layer = getattr(self, f'norm{i}')
                x_out1 = norm_layer(x1)
                x_out1 = self.mlps[i](x_out1)
                outs1.append(x_out1)  # 将输出结果添加到列表out中，猜测用于后续的解码器残差连接

        return tuple(outs1)

    def forward(self, x1):
        x1 = self.forward_features(x1)
        return x1



class DCNM2(nn.Module):
    # depths：深度，决定每层的mamba数量   out_indices:控制那个阶段输出特征，这里表示每个阶段都要输出特征
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        # nn.ModuleList是PyTorch内置的一种特殊的容器，用于存储多个nn.Module子模块。
        # 这里是创建了一个列表downsample_layers,用来储存下采样层
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
        )
        # 将模型起始部分的stem层，添加到下采样层（downsample_layer）当中
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            # 将stem后的三个下采样层储存在列表downsample_layers
            self.downsample_layers.append(downsample_layer)

            # 编码器四层，为什么储存三层，猜测原因：stem和前三层要用于解码器残差连接

        # 类似上面，创建列表，分别储存各阶段的不同操作
        self.stages = nn.ModuleList()  # 主要是mamba操作
        self.gscs = nn.ModuleList()  # 主要是卷积操作
        num_slices_list = [64, 32, 16, 8]  # 指定每个阶段的分片数，用来控制MambaLayer的分块数量
        cur = 0  # 用来追踪当前阶段的特征层数目
        for i in range(4):  # i=[0,1,2,3]
            gsc = HGDC(dims[i])  # GSC只有一个参数:in_channel

            stage = nn.Sequential(
                # manba定义了两个参数：dim和num_slices,别的参数都固定了
                # *是用来重复操作，重复次数由内部for循环控制，这里depths=2，即mamba重复两次
                *[RMB(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
                # 这里需要修改，[64,64,64,]->[256,256,20]
            )

            # 上述创建的列表里面是空的，经过for循环后，可以将每层的操作储存在相应列表中
            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]  # 累计记录已经创建的特征提取层数

        self.out_indices = out_indices  # 用来指定输出阶段，便于后续模块处理

        # 同样，创建一个列表储存mlp操作
        self.mlps = nn.ModuleList()
        for i_layer in range(4):  # i_layer=[0,1,2,3]
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            # MlpChannel定义两个参数：in_channel,out_channel
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))
    def forward_features(self, x2):
        outs2 = []
        for i in range(4):  # i=[0,1,2,3]
            x2 = self.downsample_layers[i](x2)
            x2 = self.gscs[i](x2)
            x2 = self.stages[i](x2)

            # out_indices作用：列表表示要在这些位置进行归一化和mlp操作
            if i in self.out_indices:  # 判断i是否在列表out_indices=[0,1,2,3]中
                # getattr用于动态获取归一化层norm[i],通过名字就可以获取该层归一化操作
                norm_layer = getattr(self, f'norm{i}')
                x_out2 = norm_layer(x2)
                x_out2 = self.mlps[i](x_out2)
                outs2.append(x_out2)  # 将输出结果添加到列表out中，猜测用于后续的解码器残差连接

        return tuple(outs2)

    def forward(self, x2):
        x2 = self.forward_features(x2)
        return x2


class DCNM3(nn.Module):
    # depths：深度，决定每层的mamba数量   out_indices:控制那个阶段输出特征，这里表示每个阶段都要输出特征
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        # nn.ModuleList是PyTorch内置的一种特殊的容器，用于存储多个nn.Module子模块。
        # 这里是创建了一个列表downsample_layers,用来储存下采样层
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
        )
        # 将模型起始部分的stem层，添加到下采样层（downsample_layer）当中
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            # 将stem后的三个下采样层储存在列表downsample_layers
            self.downsample_layers.append(downsample_layer)

            # 编码器四层，为什么储存三层，猜测原因：stem和前三层要用于解码器残差连接

        # 类似上面，创建列表，分别储存各阶段的不同操作
        self.stages = nn.ModuleList()  # 主要是mamba操作
        self.gscs = nn.ModuleList()  # 主要是卷积操作
        num_slices_list = [64, 32, 16, 8]  # 指定每个阶段的分片数，用来控制MambaLayer的分块数量
        cur = 0  # 用来追踪当前阶段的特征层数目
        for i in range(4):  # i=[0,1,2,3]
            gsc = HGDC(dims[i])  # GSC只有一个参数:in_channel

            stage = nn.Sequential(
                # manba定义了两个参数：dim和num_slices,别的参数都固定了
                # *是用来重复操作，重复次数由内部for循环控制，这里depths=2，即mamba重复两次
                *[RMB(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
                # 这里需要修改，[64,64,64,]->[256,256,20]
            )

            # 上述创建的列表里面是空的，经过for循环后，可以将每层的操作储存在相应列表中
            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]  # 累计记录已经创建的特征提取层数

        self.out_indices = out_indices  # 用来指定输出阶段，便于后续模块处理

        # 同样，创建一个列表储存mlp操作
        self.mlps = nn.ModuleList()
        for i_layer in range(4):  # i_layer=[0,1,2,3]
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            # MlpChannel定义两个参数：in_channel,out_channel
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))
    def forward_features(self, x3):
        outs3 = []
        for i in range(4):  # i=[0,1,2,3]
            x3 = self.downsample_layers[i](x3)
            x3 = self.gscs[i](x3)
            x3 = self.stages[i](x3)

            # out_indices作用：列表表示要在这些位置进行归一化和mlp操作
            if i in self.out_indices:  # 判断i是否在列表out_indices=[0,1,2,3]中
                # getattr用于动态获取归一化层norm[i],通过名字就可以获取该层归一化操作
                norm_layer = getattr(self, f'norm{i}')
                x_out3 = norm_layer(x3)
                x_out3 = self.mlps[i](x_out3)
                outs3.append(x_out3)  # 将输出结果添加到列表out中，猜测用于后续的解码器残差连接

        return tuple(outs3)

    def forward(self, x3):
        x3 = self.forward_features(x3)
        return x3


class DCNM4(nn.Module):
    # depths：深度，决定每层的mamba数量   out_indices:控制那个阶段输出特征，这里表示每个阶段都要输出特征
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        # nn.ModuleList是PyTorch内置的一种特殊的容器，用于存储多个nn.Module子模块。
        # 这里是创建了一个列表downsample_layers,用来储存下采样层
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
        )
        # 将模型起始部分的stem层，添加到下采样层（downsample_layer）当中
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            # 将stem后的三个下采样层储存在列表downsample_layers
            self.downsample_layers.append(downsample_layer)

            # 编码器四层，为什么储存三层，猜测原因：stem和前三层要用于解码器残差连接

        # 类似上面，创建列表，分别储存各阶段的不同操作
        self.stages = nn.ModuleList()  # 主要是mamba操作
        self.gscs = nn.ModuleList()  # 主要是卷积操作
        num_slices_list = [64, 32, 16, 8]  # 指定每个阶段的分片数，用来控制MambaLayer的分块数量
        cur = 0  # 用来追踪当前阶段的特征层数目
        for i in range(4):  # i=[0,1,2,3]
            gsc = HGDC(dims[i])  # GSC只有一个参数:in_channel

            stage = nn.Sequential(
                # manba定义了两个参数：dim和num_slices,别的参数都固定了
                # *是用来重复操作，重复次数由内部for循环控制，这里depths=2，即mamba重复两次
                *[RMB(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
                # 这里需要修改，[64,64,64,]->[256,256,20]
            )

            # 上述创建的列表里面是空的，经过for循环后，可以将每层的操作储存在相应列表中
            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]  # 累计记录已经创建的特征提取层数

        self.out_indices = out_indices  # 用来指定输出阶段，便于后续模块处理

        # 同样，创建一个列表储存mlp操作
        self.mlps = nn.ModuleList()
        for i_layer in range(4):  # i_layer=[0,1,2,3]
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            # MlpChannel定义两个参数：in_channel,out_channel
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))
    def forward_features(self, x4):
        outs4 = []
        for i in range(4):  # i=[0,1,2,3]
            x4 = self.downsample_layers[i](x4)
            x4 = self.gscs[i](x4)
            x4 = self.stages[i](x4)

            # out_indices作用：列表表示要在这些位置进行归一化和mlp操作
            if i in self.out_indices:  # 判断i是否在列表out_indices=[0,1,2,3]中
                # getattr用于动态获取归一化层norm[i],通过名字就可以获取该层归一化操作
                norm_layer = getattr(self, f'norm{i}')
                x_out4 = norm_layer(x4)
                x_out4 = self.mlps[i](x_out4)
                outs4.append(x_out4)  # 将输出结果添加到列表out中，猜测用于后续的解码器残差连接

        return tuple(outs4)

    def forward(self, x4):
        x4 = self.forward_features(x4)
        return x4


                                       # enc_hidden=(1,768,8,8,1)
class CMMBMambaSeg(nn.Module):
    def __init__(
            self,
            in_chans=1,
            out_chans=13,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size: int = 768,
            norm_name="instance",
            conv_block: bool = True,
            res_block: bool = True,
            spatial_dims=3,
            emb_dim=768
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.mq1 = nn.Linear(emb_dim, emb_dim, bias=False)
        self.mk1 = nn.Linear(emb_dim, emb_dim, bias=False)
        self.mv1 = nn.Linear(emb_dim, emb_dim, bias=False)

        self.mk2_1 = nn.Linear(emb_dim, emb_dim, bias=False)
        self.mv2_1 = nn.Linear(emb_dim, emb_dim, bias=False)
        self.mk2_2 = nn.Linear(emb_dim, emb_dim, bias=False)
        self.mv2_2 = nn.Linear(emb_dim, emb_dim, bias=False)

        self.mk3_1 = nn.Linear(emb_dim, emb_dim, bias=False)
        self.mv3_1 = nn.Linear(emb_dim, emb_dim, bias=False)
        self.mk3_2 = nn.Linear(emb_dim, emb_dim, bias=False)
        self.mv3_2 = nn.Linear(emb_dim, emb_dim, bias=False)

        self.mq4 = nn.Linear(emb_dim, emb_dim, bias=False)
        self.mk4 = nn.Linear(emb_dim, emb_dim, bias=False)
        self.mv4 = nn.Linear(emb_dim, emb_dim, bias=False)

        self.norm = nn.LayerNorm(emb_dim)
        # self.transform_matrix = nn.Linear(4 * emb_dim, emb_dim)
        self.transform_matrix = nn.Linear(8 * emb_dim, emb_dim)

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value

        self.spatial_dims = spatial_dims
        self.vit1 = DCNM1(in_chans,
                                  depths=depths,
                                  dims=feat_size,
                                  drop_path_rate=drop_path_rate,
                                  layer_scale_init_value=layer_scale_init_value,
        )
        self.vit2 = DCNM2(in_chans,
                                  depths=depths,
                                  dims=feat_size,
                                  drop_path_rate=drop_path_rate,
                                  layer_scale_init_value=layer_scale_init_value,
        )
        self.vit3 = DCNM3(in_chans,
                                  depths=depths,
                                  dims=feat_size,
                                  drop_path_rate=drop_path_rate,
                                  layer_scale_init_value=layer_scale_init_value,
        )
        self.vit4 = DCNM4(in_chans,
                                  depths=depths,
                                  dims=feat_size,
                                  drop_path_rate=drop_path_rate,
                                  layer_scale_init_value=layer_scale_init_value,
        )

        # 分支1
        self.encoder1_1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,  # 1
            out_channels=self.feat_size[0],  # 48
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2_1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],  # 48
            out_channels=self.feat_size[1],  # 96
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3_1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],  # 96
            out_channels=self.feat_size[2],  # 192
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4_1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],  # 192
            out_channels=self.feat_size[3],  # 384
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5_1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],  # 384
            out_channels=self.hidden_size,  # 768
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        # 分支2
        self.encoder1_2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,  # 1
            out_channels=self.feat_size[0],  # 48
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2_2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],  # 48
            out_channels=self.feat_size[1],  # 96
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3_2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],  # 96
            out_channels=self.feat_size[2],  # 192
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4_2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],  # 192
            out_channels=self.feat_size[3],  # 384
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5_2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],  # 384
            out_channels=self.hidden_size,  # 768
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        # 分支3
        self.encoder1_3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,  # 1
            out_channels=self.feat_size[0],  # 48
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2_3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],  # 48
            out_channels=self.feat_size[1],  # 96
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3_3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],  # 96
            out_channels=self.feat_size[2],  # 192
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4_3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],  # 192
            out_channels=self.feat_size[3],  # 384
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5_3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],  # 384
            out_channels=self.hidden_size,  # 768
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        # 分支4
        self.encoder1_4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,  # 1
            out_channels=self.feat_size[0],  # 48
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2_4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],  # 48
            out_channels=self.feat_size[1],  # 96
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3_4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],  # 96
            out_channels=self.feat_size[2],  # 192
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4_4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],  # 192
            out_channels=self.feat_size[3],  # 384
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5_4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],  # 384
            out_channels=self.hidden_size,  # 768
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        # self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=48,
        #                         out_channels=self.out_chans)  # out_channels=13
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=48,
                                out_channels=self.out_chans)  # out_channels=13
        self.last_conv = nn.Conv3d(in_channels=4, out_channels=16, kernel_size=1)
        # ini_weights(self.module_list())

    def proj_feat(self, x):
        # x.size(0)获取x的第0维度：batch_size   self.proj_view_shape: 这是一个类属性，是一个列表或元组，定义新的形状（除了batch_size）
        new_view = [x.size(0)] + self.proj_view_shape
        # view(new_view)：将张量x重塑为new_size的形状，且元素总数不变  eg:(32，48)->(32,16,3)
        x = x.view(new_view)
        # 用于张量x维度重新排序
        # self.proj_axes是列表或元组，指定了新维度的顺序   contiguous()保证张量在内存中连续
        x = x.permute(self.proj_axes).contiguous()
        return x

    # def forward(self, x_in1, x_in2, x_in3, x_in4):

    def forward(self, x):

        # 将x按照chennel分成四份
        #x1, x2, x3, x4 = np.split(x.reshape(1, 4, 128, 128, 128), 4, axis=1)
        x1, x2, x3, x4 = np.split(x.reshape(1, 4, 128, 128, 64), 4, axis=1)

        x_in1 = x1.repeat(1, 4, 1, 1, 1)
        x_in2 = x2.repeat(1, 4, 1, 1, 1)
        x_in3 = x3.repeat(1, 4, 1, 1, 1)
        x_in4 = x4.repeat(1, 4, 1, 1, 1)

        # x_in1 = x[ :, 0:1, :, :, :]
        # x_in2 = x[ :, 1:2, :, :, :]
        # x_in3 = x[ :, 2:3, :, :, :]
        # x_in4 = x[ :, 3:4, :, :, :]

        "多分枝编码器特征提取阶段"
        outs1 = self.vit1(x_in1)
        outs2 = self.vit2(x_in2)
        outs3 = self.vit3(x_in3)
        outs4 = self.vit4(x_in4)

        enc1_1 = self.encoder1_1(x_in1)
        x2_1 = outs1[0]
        enc2_1 = self.encoder2_1(x2_1)
        x3_1 = outs1[1]
        enc3_1 = self.encoder3_1(x3_1)
        x4_1 = outs1[2]
        enc4_1 = self.encoder4_1(x4_1)
        enc_hidden1 = self.encoder5_1(outs1[3])

        enc1_2 = self.encoder1_2(x_in2)
        x2_2 = outs2[0]
        enc2_2 = self.encoder2_2(x2_2)
        x3_2 = outs2[1]
        enc3_2 = self.encoder3_2(x3_2)
        x4_2 = outs2[2]
        enc4_2 = self.encoder4_2(x4_2)
        enc_hidden2 = self.encoder5_2(outs2[3])

        enc1_3 = self.encoder1_3(x_in3)
        x2_3 = outs3[0]
        enc2_3 = self.encoder2_3(x2_3)
        x3_3 = outs3[1]
        enc3_3 = self.encoder3_3(x3_3)
        x4_3 = outs3[2]
        enc4_3 = self.encoder4_3(x4_3)
        enc_hidden3 = self.encoder5_3(outs3[3])

        enc1_4 = self.encoder1_4(x_in4)
        x2_4 = outs4[0]
        enc2_4 = self.encoder2_4(x2_4)
        x3_4 = outs4[1]
        enc3_4 = self.encoder3_4(x3_4)
        x4_4 = outs4[2]
        enc4_4 = self.encoder4_4(x4_4)
        enc_hidden4 = self.encoder5_4(outs4[3])

        "跨膜态交叉注意力阶段"
        # 获取跨膜太输入特征张量的维度
        batch_size1, channels1, depth1, height1, width1 = enc_hidden1.size()
        batch_size2, channels2, depth2, height2, width2 = enc_hidden2.size()
        batch_size3, channels3, depth3, height3, width3 = enc_hidden3.size()
        batch_size4, channels4, depth4, height4, width4 = enc_hidden4.size()

        # 将空间维度展平为序列维度：(b,c,d,h.w)->(b,c,d*h*w)
        enc_hidden1 = enc_hidden1.view(batch_size1, channels1, -1)
        enc_hidden2 = enc_hidden2.view(batch_size2, channels2, -1)
        enc_hidden3 = enc_hidden3.view(batch_size3, channels3, -1)
        enc_hidden4 = enc_hidden4.view(batch_size4, channels4, -1)

        enc_hidden1 = enc_hidden1.permute(0, 2, 1)       # (b,c,d*h*w)->(b,d*h*w,c)
        enc_hidden2 = enc_hidden2.permute(0, 2, 1)
        enc_hidden3 = enc_hidden3.permute(0, 2, 1)
        enc_hidden4 = enc_hidden4.permute(0, 2, 1)


        q1 = self.mq1(enc_hidden1)    # (b,seq_len,c )
        k2_1 = self.mk2_1(enc_hidden2)
        v2_1 = self.mv2_1(enc_hidden2)
        k3_1 = self.mk3_1(enc_hidden3)
        v3_1 = self.mv3_1(enc_hidden3)
        k4 = self.mk4(enc_hidden4)
        v4 = self.mv4(enc_hidden4)

        q4 = self.mq4(enc_hidden4)
        k2_2 = self.mk2_2(enc_hidden2)
        v2_2 = self.mv2_2(enc_hidden2)
        k3_2 = self.mk3_2(enc_hidden3)
        v3_2 = self.mv3_2(enc_hidden3)
        k1 = self.mk1(enc_hidden1)
        v1 = self.mv1(enc_hidden1)


        att2_1 = torch.matmul(q1, k2_1.transpose(-2, -1)) / np.sqrt(self.emb_dim)  # (b, seq_len, seq_len)
        att2_1 = torch.softmax(att2_1, dim=-1)                                     # (b, seq_len, seq_len)
        att3_1 = torch.matmul(q1, k3_1.transpose(-2, -1)) / np.sqrt(self.emb_dim)
        att3_1 = torch.softmax(att3_1, dim=-1)
        att4_1 = torch.matmul(q1, k4.transpose(-2, -1)) / np.sqrt(self.emb_dim)
        att4_1 = torch.softmax(att4_1, dim=-1)

        att2_2 = torch.matmul(q4, k2_2.transpose(-2, -1)) / np.sqrt(self.emb_dim)
        att2_2 = torch.softmax(att2_2, dim=-1)
        att3_2 = torch.matmul(q4, k3_2.transpose(-2, -1)) / np.sqrt(self.emb_dim)
        att3_2 = torch.softmax(att3_2, dim=-1)
        att4_2 = torch.matmul(q4, k1.transpose(-2, -1)) / np.sqrt(self.emb_dim)
        att4_2 = torch.softmax(att4_2, dim=-1)


        q1 = q1.transpose(1, 2).squeeze(0).transpose(0,1)                  # (b, c, seq_len)
        q1 = self.norm(q1)
        out2_1 = torch.matmul(att2_1, v2_1)             # (b, seq_len, c)
        out2_1 = out2_1.transpose(1, 2).squeeze(0).transpose(0,1)         # (b, c, seq_len)
        out2_1 = self.norm(out2_1)
        out3_1 = torch.matmul(att3_1, v3_1)
        out3_1 = out3_1.transpose(1, 2).squeeze(0).transpose(0,1)
        out3_1 = self.norm(out3_1)
        out4_1 = torch.matmul(att4_1, v4)
        out4_1 = out4_1.transpose(1, 2).squeeze(0).transpose(0,1)
        out4_1 = self.norm(out4_1)

        q4 = q4.transpose(1, 2).squeeze(0).transpose(0,1)
        q4 = self.norm(q4)
        out2_2 = torch.matmul(att2_2, v2_2)
        out2_2 = out2_2.transpose(1, 2).squeeze(0).transpose(0,1)
        out2_2 = self.norm(out2_2)
        out3_2 = torch.matmul(att3_2, v3_2)
        out3_2 = out3_2.transpose(1, 2).squeeze(0).transpose(0,1)
        out3_2 = self.norm(out3_2)
        out4_2 = torch.matmul(att4_2, v4)
        out4_2 = out4_2.transpose(1, 2).squeeze(0).transpose(0,1)
        out4_2 = self.norm(out4_2)


        out_1 = torch.cat((q1, out2_1, out3_1, out4_1), dim=1)  # dim=1,在channel维度上做拼接
        out_2 = torch.cat((q4, out2_2, out3_2, out4_2), dim=1)

        out = torch.cat((out_1, out_2), dim=1)

        enc_hidden = self.transform_matrix(out)                # 做维度映射：(b,c,4*d*h*w)->(b,c,d*h*w)
        # enc_hidden = enc_hidden.view(batch_size1, 768, 16, 16, 1)
        enc_hidden = enc_hidden.view(1, 768, 8, 8, 4)            #[1,768,8,8,16]
        # enc_hidden = enc_hidden.view(1, 768, 8, 8, 4)  # [1,768,8,8,16]

        " 注意这里还需要修改解码器残差输入。每层解码器需要接受四个残差 "

        dec3 = self.decoder5(enc_hidden, enc4_1, enc4_2, enc4_3, enc4_4)#torch.Size([1, 384, 16, 16, 16])
        dec2 = self.decoder4(dec3, enc3_1, enc3_2, enc3_3, enc3_4)
        dec1 = self.decoder3(dec2, enc2_1, enc2_2, enc2_3, enc2_4)
        dec0 = self.decoder2(dec1, enc1_1, enc1_2, enc1_3, enc1_4)
        out = self.decoder1(dec0)
        out = self.out(out)

        return out

        # return self.last_conv(out)
