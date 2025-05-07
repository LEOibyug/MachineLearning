import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import os
import matplotlib.pyplot as plt

# 设备配置，支持 GPU 加速
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义自注意力层，用于增强生成器的特征提取能力
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

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

# 定义残差块，用于生成器网络
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        """
        前向传播函数，计算残差块输出
        :param x: 输入特征
        :return: 残差块输出，输入与卷积块输出的和
        """
        return x + self.conv_block(x)

# 定义生成器网络，将自注意力机制加入到架构中
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()
        # 初始卷积层
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # 下采样层
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
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
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # 输出层
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """
        前向传播函数，生成风格转换后的图像
        :param x: 输入图像张量
        :return: 风格转换后的图像张量
        """
        return self.model(x)

# 定义判别器网络，用于区分真实图像和生成图像
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """
        前向传播函数，判断图像真实性
        :param x: 输入图像张量
        :return: 判别结果，值越接近1表示越真实
        """
        x = self.model(x)
        return x

# 定义数据加载器和预处理逻辑
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化处理
])

# 加载真实图像和莫奈风格图像数据集
real_dataset = ImageFolder(root='./pics/train/real/', transform=transform)
monet_dataset = ImageFolder(root='./pics/train/monet/', transform=transform)

real_loader = DataLoader(real_dataset, batch_size=1, shuffle=True)
monet_loader = DataLoader(monet_dataset, batch_size=1, shuffle=True)

# 初始化生成器和判别器模型
G_R2M = Generator(3, 3).to(DEVICE)  # 真实到莫奈的生成器
G_M2R = Generator(3, 3).to(DEVICE)  # 莫奈到真实的生成器
D_R = Discriminator(3).to(DEVICE)    # 真实图像判别器
D_M = Discriminator(3).to(DEVICE)    # 莫奈图像判别器

# 定义损失函数
criterion_GAN = nn.MSELoss()  # GAN 损失，用于对抗训练
criterion_cycle = nn.L1Loss()  # 循环一致性损失
criterion_identity = nn.L1Loss()  # 身份映射损失

# 定义优化器
optimizer_G = optim.Adam(list(G_R2M.parameters()) + list(G_M2R.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_R = optim.Adam(D_R.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_M = optim.Adam(D_M.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 定义图像反归一化函数，用于可视化
def denorm(img_tensors):
    return img_tensors * 0.5 + 0.5

# 训练循环
epochs = 400
for epoch in range(epochs):
    for i, (real_batch, monet_batch) in enumerate(zip(real_loader, monet_loader)):
        real_imgs = real_batch[0].to(DEVICE)
        monet_imgs = monet_batch[0].to(DEVICE)

        # 训练判别器
        optimizer_D_R.zero_grad()
        optimizer_D_M.zero_grad()

        fake_monet = G_R2M(real_imgs)
        fake_real = G_M2R(monet_imgs)

        # 真实图像和生成图像的判别结果
        real_validity = D_R(real_imgs)
        fake_validity = D_R(fake_real.detach())
        D_R_loss = (criterion_GAN(real_validity, torch.ones_like(real_validity)) + criterion_GAN(fake_validity, torch.zeros_like(fake_validity))) / 2
        D_R_loss.backward()
        optimizer_D_R.step()

        real_validity = D_M(monet_imgs)
        fake_validity = D_M(fake_monet.detach())
        D_M_loss = (criterion_GAN(real_validity, torch.ones_like(real_validity)) + criterion_GAN(fake_validity, torch.zeros_like(fake_validity))) / 2
        D_M_loss.backward()
        optimizer_D_M.step()

        # 训练生成器
        optimizer_G.zero_grad()

        fake_monet = G_R2M(real_imgs)
        fake_real = G_M2R(monet_imgs)

        recov_real = G_M2R(fake_monet)
        recov_monet = G_R2M(fake_real)

        # 计算生成器损失，包括 GAN 损失、循环一致性损失和身份映射损失
        gan_loss_R2M = criterion_GAN(D_M(fake_monet), torch.ones_like(D_M(fake_monet)))
        gan_loss_M2R = criterion_GAN(D_R(fake_real), torch.ones_like(D_R(fake_real)))
        cycle_loss_R = criterion_cycle(recov_real, real_imgs)
        cycle_loss_M = criterion_cycle(recov_monet, monet_imgs)
        identity_loss_R = criterion_identity(G_M2R(real_imgs), real_imgs)
        identity_loss_M = criterion_identity(G_R2M(monet_imgs), monet_imgs)

        G_loss = gan_loss_R2M + gan_loss_M2R + 10.0 * (cycle_loss_R + cycle_loss_M) + 5.0 * (identity_loss_R + identity_loss_M)
        G_loss.backward()
        optimizer_G.step()

        if i % 50 == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}] [D_R loss: {D_R_loss.item():.4f}] [D_M loss: {D_M_loss.item():.4f}] [G loss: {G_loss.item():.4f}]")

        # 每 200 个批次保存一次图像结果
        if i % 200 == 0:
            with torch.no_grad():
                fake_monet_ = denorm(fake_monet).cpu().permute(0, 2, 3, 1).numpy()[0]
                real_imgs_ = denorm(real_imgs).cpu().permute(0, 2, 3, 1).numpy()[0]
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(real_imgs_)
                plt.title('Real Image')
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(fake_monet_)
                plt.title('Generated Monet')
                plt.axis('off')
                plt.savefig(f'./pics/output/epoch_{epoch}_batch_{i}.png')
                plt.close()

    # 每 10 个 epoch 保存一次模型
    if (epoch + 1) % 10 == 0:
        torch.save(G_R2M.state_dict(), f'./models/G_R2M/generator_{epoch+1}.pth')
        torch.save(G_M2R.state_dict(), f'./models/G_M2R/generator_{epoch+1}.pth')
        torch.save(D_R.state_dict(), f'./models/dis/discriminator_R_{epoch+1}.pth')
        torch.save(D_M.state_dict(), f'./models/dis/discriminator_M_{epoch+1}.pth')
        print(f"Models saved at epoch {epoch+1}")
