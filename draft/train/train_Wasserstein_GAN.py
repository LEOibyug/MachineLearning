import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
import itertools
import matplotlib.pyplot as plt
from PIL import Image
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'using device {device}')

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        attention = torch.bmm(query, key)
        attention = torch.softmax(attention, dim=-1)
        value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

class Generator(nn.Module):
    def __init__(self, num_residual_blocks=9):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.downsampling = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_residual_blocks // 2)],
            SelfAttention(256),
            *[ResidualBlock(256) for _ in range(num_residual_blocks // 2)]
        )
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.downsampling(x)
        x = self.residual_blocks(x)
        x = self.upsampling(x)
        return self.output(x)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

train_preprocess = transforms.Compose([
    transforms.Resize((286, 286)),  # 首先将图像放大
    transforms.RandomCrop((256, 256)),  # 然后随机裁剪为256x256
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 颜色抖动
    AddGaussianNoise(mean=0.0, std=0.1),  # 随机加入一些高斯噪声
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])

def image_loader(path):
    return Image.open(path).convert('RGB')

data_base_real = DatasetFolder('../gan-getting-started/real', loader=image_loader, extensions=('jpg', 'jpeg', 'png'),
                               transform=train_preprocess)
data_base_monet = DatasetFolder('../gan-getting-started/monet_jpg', loader=image_loader,
                                extensions=('jpg', 'jpeg', 'png'),
                                transform=train_preprocess)
batch_size = 15

dataloader_real = DataLoader(data_base_real, batch_size=batch_size, shuffle=True)
dataloader_monet = DataLoader(data_base_monet, batch_size=batch_size, shuffle=True)

G_R2M = Generator().to(device)
G_M2R = Generator().to(device)
C_R = Critic().to(device)
C_M = Critic().to(device)

optimizer_G = optim.Adam(itertools.chain(G_R2M.parameters(), G_M2R.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_C_R = optim.Adam(C_R.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_C_M = optim.Adam(C_M.parameters(), lr=0.0002, betas=(0.5, 0.999))

epoch_num = 600
n_critic = 5  # Number of critic iterations per generator iteration
clip_value = 0.01  # Weight clipping value

def train(dataloader_real, dataloader_monet, G_R2M, G_M2R, C_R, C_M, optimizer_G, optimizer_C_R, optimizer_C_M, epoch_num):
    for epoch in range(epoch_num):
        for i, (real_images, monet_images) in enumerate(zip(dataloader_real, dataloader_monet)):
            real_images = real_images[0].to(device)
            monet_images = monet_images[0].to(device)

            # Train Critic
            for _ in range(n_critic):
                optimizer_C_R.zero_grad()
                optimizer_C_M.zero_grad()

                fake_monet = G_R2M(real_images).detach()
                fake_real = G_M2R(monet_images).detach()

                # Critic loss
                loss_C_R = -(torch.mean(C_R(real_images)) - torch.mean(C_R(fake_real)))
                loss_C_M = -(torch.mean(C_M(monet_images)) - torch.mean(C_M(fake_monet)))

                loss_C_R.backward()
                loss_C_M.backward()
                optimizer_C_R.step()
                optimizer_C_M.step()

                # Weight clipping
                for p in C_R.parameters():
                    p.data.clamp_(-clip_value, clip_value)
                for p in C_M.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            # Train Generator
            optimizer_G.zero_grad()

            fake_monet = G_R2M(real_images)
            fake_real = G_M2R(monet_images)

            # Generator loss
            loss_G_R2M = -torch.mean(C_M(fake_monet))
            loss_G_M2R = -torch.mean(C_R(fake_real))

            rec_real = G_M2R(fake_monet)
            loss_cycle_real = nn.functional.l1_loss(rec_real, real_images)

            rec_monet = G_R2M(fake_real)
            loss_cycle_monet = nn.functional.l1_loss(rec_monet, monet_images)

            loss_G = loss_G_R2M + loss_G_M2R + 10.0 * (loss_cycle_real + loss_cycle_monet)
            loss_G.backward()
            optimizer_G.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{epoch_num}], Step [{i}/{len(dataloader_real)}], "
                      f"Loss_G: {loss_G.item():.4f}, Loss_C_R: {loss_C_R.item():.4f}, Loss_C_M: {loss_C_M.item():.4f}")

        if epoch % 20 == 0:
            torch.save(G_R2M.state_dict(), f'G_R2M_Wasserstein_GAN_epoch_{epoch}.pth')

train(dataloader_real, dataloader_monet, G_R2M, G_M2R, C_R, C_M, optimizer_G, optimizer_C_R, optimizer_C_M, epoch_num)