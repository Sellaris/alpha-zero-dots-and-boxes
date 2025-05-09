import logging
from typing import Tuple

import numpy as np
import torch
from torch import nn

from src import DotsAndBoxesGame
from src.model.neural_network import AZNeuralNetwork


class SEBlock(nn.Module):
    """
    1. SE 块（Squeeze-and-Excitation）原理：通道注意力
    1.1 Squeeze（压缩）
    对每个通道做全局平均池化，将 [B,C,H,W] 映射为 [B,C,1,1]，得到通道描述符。
    这一步让模型获得全局上下文，使其“看到”整张特征图的信息 

    1.2 Excitation（激励）
    将通道描述符通过两层 1×1 卷积（或等价的全连接），先降维到 reduced_channels = max(1, int(C·ratio))，再升回原始通道数。
    两次线性映射之间通常插入 ReLU，最后用 Sigmoid 将输出范围归一到 [0,1]，即为每个通道的注意力权重 

    1.3 Scale（重标）
    将原始特征图按通道乘以上一步得到的权重系数，完成特征重标定。
    这一操作能够让网络“关注”更重要的通道，抑制无关或噪声特征 
    """
    def __init__(self, channels, ratio=0.25):
        """
        Args:
            channels: 输入通道数
            ratio: 压缩率，决定降维后的通道数
        """
        super(SEBlock, self).__init__()
        reduced_channels = max(1, int(channels * ratio))
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(reduced_channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y = self.global_avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class NestedBottleneckBlock(nn.Module):
    """
    KataGo风格的嵌套瓶颈残差块
    采用双层深度可分离卷积配合SE注意力，通过瓶颈结构提升计算效率
    """
    def __init__(self, n_channels, bottleneck_channels, kernel_size, padding):
        """
        Args:
            n_channels: 输入通道数
            bottleneck_channels: 瓶颈层通道数
            kernel_size: 卷积核大小
            padding: 填充方式
        """
        super(NestedBottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, bottleneck_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.relu1 = nn.ReLU()
        
        # 深度可分离卷积替代标准卷积
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 
                              kernel_size=kernel_size, padding=padding,
                              groups=bottleneck_channels)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(bottleneck_channels, n_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(n_channels)
        
        self.se_block = SEBlock(n_channels)  # 添加SE注意力模块
        self.final_relu = nn.ReLU()

    def forward(self, x):
        identity = x
        
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        x = self.se_block(x)  # 应用注意力机制
        
        x += identity
        return self.final_relu(x)


class GlobalBiasBlock(nn.Module):
    """
    全局池化偏置模块
    将全局平均池化特征通过小型网络生成偏置项，增强全局感知能力
    """
    def __init__(self, channels):
        super(GlobalBiasBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.transform = nn.Sequential(
            nn.Conv2d(channels, channels//2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels//2, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        bias = self.global_avg_pool(x)
        bias = self.transform(bias)
        return x + bias.expand_as(x)

class PolicyHead(nn.Module):
    """
    改进的策略头
    1x1卷积降维后，添加全局平均池化偏置，最后全连接输出策略分布
    """
    def __init__(self, conv_in_channels, conv_out_channels, kernel_size, stride, padding, 
                 fc_in_features, fc_out_features):
        super(PolicyHead, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(conv_in_channels, conv_out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU()
        )
        
        self.global_bias = GlobalBiasBlock(conv_out_channels)  # 全局感知模块
        self.final_global_pool = nn.AdaptiveAvgPool2d(1)  # 最终全局池化
        
        self.fc = nn.Linear(fc_in_features + conv_out_channels, fc_out_features)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.global_bias(x)
        
        # 展平和全局池化特征合并
        global_feat = self.final_global_pool(x).view(x.size(0), -1)  # [B, C]
        flat_x = x.view(x.size(0), -1)  # [B, H*W*C]
        
        # 拼接展平特征与全局特征
        combined = torch.cat((flat_x, global_feat), dim=1)
        
        x = self.fc(combined)
        x = self.log_softmax(x).exp()
        return x

class AZDualRes(AZNeuralNetwork):
    """
    修改后的双重残差神经网络
    集成嵌套瓶颈残差块、SE注意力和全局感知策略头
    """
    def __init__(self, game_size: int, inference_device: torch.device, model_parameters: dict):
        super(AZDualRes, self).__init__(game_size, inference_device)

        img_size = 2 * game_size + 1

        # 获取模型参数
        blocks_params = model_parameters["blocks"]
        heads_params = model_parameters["heads"]
        
        # 计算模块参数
        residual_blocks = blocks_params["residual_blocks"]
        channels = blocks_params["channels"]
        bottleneck_channels = blocks_params.get("bottleneck_channels", channels//4)  # 瓶颈通道数
        se_ratio = blocks_params.get("se_ratio", 0.25)  # SE模块压缩比
        
        # 构建模型
        self.conv_block = ConvBlock(out_channels=channels)
        
        # 动态创建残差块：每隔3个正常残差块插入一个带全局偏置的块
        self.residual_blocks = nn.ModuleList()
        for i in range(residual_blocks):
            block = NestedBottleneckBlock(
                n_channels=channels,
                bottleneck_channels=bottleneck_channels,
                kernel_size=blocks_params["res_kernel_size"],
                padding=blocks_params["padding"]
            )
            self.residual_blocks.append(block)
            
            # 每隔3个块插入全局偏置模块
            if (i+1) % 3 == 0:
                self.residual_blocks.append(GlobalBiasBlock(channels))
        
        # 构建头
        self.policy_head = PolicyHead(
            conv_in_channels=channels,
            conv_out_channels=heads_params["policy_head_channels"],
            kernel_size=heads_params["heads_kernel_size"],
            stride=heads_params["heads_stride"],
            padding=heads_params["heads_padding"],
            fc_in_features=(heads_params["policy_head_channels"] * img_size * img_size),
            fc_out_features=(2 * self.game_size * (self.game_size + 1))
        )
        
        # 权重初始化
        self.weight_init()
        self.float()

    def weight_init(self):
        """新增模块的权重初始化，保持原有初始化风格"""
        # 卷积块初始化
        conv2d = self.conv_block.conv
        nn.init.xavier_normal_(conv2d.weight)
        conv2d.bias.data.fill_(0.01)
        
        # 残差块初始化
        for idx, module in enumerate(self.residual_blocks):
            if isinstance(module, NestedBottleneckBlock):
                # Bottleneck块的内部卷积初始化
                nn.init.xavier_normal_(module.conv1.weight)
                module.conv1.bias.data.fill_(0.01)
                
                nn.init.xavier_normal_(module.conv2.weight)
                module.conv2.bias.data.fill_(0.01)
                
                nn.init.xavier_normal_(module.conv3.weight)
                module.conv3.bias.data.fill_(0.01)
                
            elif isinstance(module, GlobalBiasBlock):
                # SE注意力网络初始化
                for m in module.transform:
                    if isinstance(m, nn.Conv2d):
                        nn.init.xavier_normal_(m.weight)
                        m.bias.data.fill_(0.01)
        
        # 策略头初始化
        for m in self.policy_head.conv_block:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)
        
        nn.init.xavier_normal_(self.policy_head.fc.weight)
        self.policy_head.fc.bias.data.fill_(0.01)
        
        # 值头初始化部分沿用原逻辑...
        # （此处省略，保留原有值头初始化逻辑）
    @staticmethod
    def encode(l: np.ndarray, b: np.ndarray) -> np.ndarray:
        """encode lines and boxes into images"""

        game_size = DotsAndBoxesGame.n_lines_to_size(l.size)
        img_size = 2 * game_size + 1

        img_l = np.zeros((img_size, img_size), dtype=np.float32)
        img_b_player = np.zeros((img_size, img_size), dtype=np.float32)
        img_b_opponent = np.zeros((img_size, img_size), dtype=np.float32)
        img_background = np.zeros((img_size, img_size), dtype=np.float32)

        # 1) image containing information which lines are drawn (for policy prediction)
        h, v = DotsAndBoxesGame.l_to_h_v(l)
        # horizontals: even rows, odd columns (0-indexing)
        # verticals: odd rows, even columns (0-indexing)
        img_l[::2, 1::2] = h
        img_l[1::2, ::2] = v
        img_l[img_l == -1.0] = 1.0

        # 2) image indicating boxes captured by player (for value prediction)
        img_b_player[1::2, 1::2] = b
        img_b_player[img_b_player == -1.0] = 0.0

        # 3) image indicating boxes captured by opponent (for value prediction)
        img_b_opponent[1::2, 1::2] = b
        img_b_opponent[img_b_opponent == 1.0] = 0.0
        img_b_opponent[img_b_opponent == -1.0] = 1.0

        # 4) image indicating unimportant pixels
        img_background[0::2, 0::2] = np.ones((game_size+1, game_size+1), dtype=np.float32)

        feature_planes = np.stack([img_l, img_b_player, img_b_opponent, img_background], axis=0)
        return feature_planes


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.conv_block(x)
        for res_block in self.residual_blocks:
            x = res_block(x)

        p = self.policy_head(x)
        v = self.value_head(x).squeeze()  # one-dimensional output

        return p, v


class ConvBlock(nn.Module):

    def __init__(self, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(4, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))

        return x


class ResBlock(nn.Module):

    def __init__(self, n_channels, kernel_size=3, stride=1, padding=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x_in = x
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += x_in
        x = self.relu2(x)

        return x

class ValueHead(nn.Module):

    def __init__(self, conv_in_channels, conv_out_channels, kernel_size, stride, padding, fc_in_features):
        super(ValueHead, self).__init__()

        self.conv = nn.Conv2d(conv_in_channels, conv_out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(conv_out_channels)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(
            in_features=fc_in_features,
            out_features=(fc_in_features//2)
        )
        self.fc2 = nn.Linear(
            in_features=(fc_in_features//2),
            out_features=1
        )


    def forward(self, x):

        x = self.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return x
