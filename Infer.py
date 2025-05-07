# -*- coding: utf-8 -*-
"""
Infer.py - 图像风格转换推断脚本

此脚本用于加载预训练的生成器模型，将真实图像转换为莫奈风格图像，并可视化结果。
支持随机选择图像或指定图像进行转换。
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import random

# 设备配置，支持 GPU 加速
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像转换设置，用于预处理输入图像
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小为 256x256
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化处理
])

# 定义自注意力层，用于增强生成器的特征提取能力
class SelfAttention(torch.nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = torch.nn.Parameter(torch.zeros(1))
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        """
        前向传播函数，计算自注意力机制
        :param x: 输入特征图
        :return: 加入注意力机制后的特征图
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x
        return out

# 定义残差块，用于生成器网络中的特征学习
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_features, in_features, 3),
            torch.nn.InstanceNorm2d(in_features),
            torch.nn.ReLU(inplace=True),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_features, in_features, 3),
            torch.nn.InstanceNorm2d(in_features)
        ]
        self.conv_block = torch.nn.Sequential(*conv_block)

    def forward(self, x):
        """
        前向传播函数，计算残差块输出
        :param x: 输入特征
        :return: 残差块输出，输入与卷积块输出的和
        """
        return x + self.conv_block(x)

# 定义生成器网络，将自注意力机制加入到架构中，用于图像风格转换
class Generator(torch.nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()
        # 初始卷积层
        model = [
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(input_nc, 64, 7),
            torch.nn.InstanceNorm2d(64),
            torch.nn.ReLU(inplace=True)
        ]

        # 下采样层
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                torch.nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                torch.nn.InstanceNorm2d(out_features),
                torch.nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # 添加自注意力层
        model += [SelfAttention(in_features)]

        # 残差块
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # 上采样层
        out_features = in_features // 2
        for _ in range(2):
            model += [
                torch.nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                torch.nn.InstanceNorm2d(out_features),
                torch.nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # 输出层
        model += [
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(64, output_nc, 7),
            torch.nn.Tanh()
        ]

        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        """
        前向传播函数，生成风格转换后的图像
        :param x: 输入图像张量
        :return: 风格转换后的图像张量
        """
        return self.model(x)

# 加载预训练模型
G_R2M = Generator(3, 3)  # 真实图像到莫奈风格图像的生成器
G_R2M.load_state_dict(torch.load("./models/G_R2M/GeneratorAttO1_400.pth", map_location=DEVICE))
G_R2M.to(DEVICE)
G_R2M.eval()

# 定义图像反归一化函数，用于可视化
inv_normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])

# 随机选择测试图像
random_choice = True  # 是否随机选择图像
image_path = None
if random_choice:
    test_folder = './pics/test/real/'
    image_files = [f for f in os.listdir(test_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if image_files:
        image_path = os.path.join(test_folder, random.choice(image_files))
        print(f"随机选择的图像: {image_path}")
    else:
        print("测试文件夹中没有找到图像")
else:
    image_path = './pics/test/real/1.png'  # 指定图像路径

if image_path and os.path.exists(image_path):
    # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # 使用模型进行推断
    with torch.no_grad():
        fake_monet = G_R2M(input_tensor)

    # 反归一化以便可视化
    fake_monet = inv_normalize(fake_monet.squeeze(0)).permute(1, 2, 0).cpu().numpy()
    input_image = inv_normalize(input_tensor.squeeze(0)).permute(1, 2, 0).cpu().numpy()

    # 确保值在 [0, 1] 范围内
    fake_monet = (fake_monet + 1) / 2
    input_image = (input_image + 1) / 2

    # 可视化结果
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(input_image)
    plt.title('输入图像 (真实)')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(fake_monet)
    plt.title('输出图像 (莫奈风格)')
    plt.axis('off')

    plt.show()
else:
    print(f"图像路径不存在: {image_path}")
