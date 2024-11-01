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
print(f'Using device {device}')

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

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
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


train_preprocess = transforms.Compose([
    transforms.Resize((286, 286)),  # Resize to 286x286
    transforms.RandomCrop((256, 256)),  # Randomly crop to 256x256
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Color jitter
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Image Loader
def image_loader(path):
    return Image.open(path).convert('RGB')

# Dataset and DataLoader
data_base_real = DatasetFolder('../../pics/train/Real', loader=image_loader, extensions=('jpg', 'jpeg', 'png'),
                               transform=train_preprocess)
data_base_monet = DatasetFolder('../../pics/train/Monet', loader=image_loader,
                                extensions=('jpg', 'jpeg', 'png'),
                                transform=train_preprocess)
batch_size = 20

dataloader_real = DataLoader(data_base_real, batch_size=batch_size, shuffle=True)
dataloader_monet = DataLoader(data_base_monet, batch_size=batch_size, shuffle=True)

G_R2M = Generator().to(device)
G_M2R = Generator().to(device)
D_R = Discriminator().to(device)
D_M = Discriminator().to(device)

criterion_GAN = nn.MSELoss().to(device)
criterion_cycle = nn.L1Loss().to(device)
criterion_identity = nn.L1Loss().to(device)

optimizer_G = optim.Adam(itertools.chain(G_R2M.parameters(), G_M2R.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_R = optim.Adam(D_R.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_M = optim.Adam(D_M.parameters(), lr=0.0002, betas=(0.5, 0.999))

lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100)
lr_scheduler_D_R = optim.lr_scheduler.LambdaLR(optimizer_D_R, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100)
lr_scheduler_D_M = optim.lr_scheduler.LambdaLR(optimizer_D_M, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100)

epoch_num = 200

# Training Function
def train(dataloader_real, dataloader_monet, G_R2M, G_M2R, D_R, D_M, criterion_GAN, criterion_cycle, criterion_identity, optimizer_G,
          optimizer_D_R, optimizer_D_M, lr_scheduler_G, lr_scheduler_D_R, lr_scheduler_D_M, epoch_num):
    for epoch in range(epoch_num):
        for i, (real_images, monet_images) in enumerate(zip(dataloader_real, dataloader_monet)):
            real_images = real_images[0].to(device)
            monet_images = monet_images[0].to(device)

            # Train Generators
            optimizer_G.zero_grad()

            # Identity Loss
            loss_identity_R2M = criterion_identity(G_R2M(monet_images), monet_images)
            loss_identity_M2R = criterion_identity(G_M2R(real_images), real_images)

            # GAN Loss
            fake_monet = G_R2M(real_images)
            pred_fake = D_M(fake_monet)
            loss_GAN_R2M = criterion_GAN(pred_fake, torch.ones_like(pred_fake).to(device))

            fake_real = G_M2R(monet_images)
            pred_fake = D_R(fake_real)
            loss_GAN_M2R = criterion_GAN(pred_fake, torch.ones_like(pred_fake).to(device))

            # Cycle Loss
            rec_real = G_M2R(fake_monet)
            loss_cycle_real = criterion_cycle(rec_real, real_images)

            rec_monet = G_R2M(fake_real)
            loss_cycle_monet = criterion_cycle(rec_monet, monet_images)

            # Total Generator Loss
            loss_G = (loss_GAN_R2M + loss_GAN_M2R) + 10.0 * (loss_cycle_real + loss_cycle_monet) + 5.0 * (loss_identity_R2M + loss_identity_M2R)
            loss_G.backward()
            optimizer_G.step()

            # Train Discriminator R
            optimizer_D_R.zero_grad()

            pred_real = D_R(real_images)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real).to(device))

            pred_fake = D_R(fake_real.detach())
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake).to(device))

            loss_D_R_total = (loss_D_real + loss_D_fake) * 0.5
            loss_D_R_total.backward()
            optimizer_D_R.step()

            # Train Discriminator M
            optimizer_D_M.zero_grad()

            pred_real = D_M(monet_images)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real).to(device))

            pred_fake = D_M(fake_monet.detach())
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake).to(device))

            loss_D_M_total = (loss_D_real + loss_D_fake) * 0.5
            loss_D_M_total.backward()
            optimizer_D_M.step()

            if i % 5 == 0:
                print(f"Epoch [{epoch}/{epoch_num}], Step [{i}/{len(dataloader_real)}], "
                      f"Loss_G: {loss_G.item():.4f}, Loss_D_R: {loss_D_R_total.item():.4f}, Loss_D_M: {loss_D_M_total.item():.4f}")

        lr_scheduler_G.step()
        lr_scheduler_D_R.step()
        lr_scheduler_D_M.step()

        if epoch % 20 == 0:
            torch.save(G_R2M.state_dict(), f'G_R2M_epoch_{epoch}.pth')

train(dataloader_real, dataloader_monet, G_R2M, G_M2R, D_R, D_M, criterion_GAN, criterion_cycle, criterion_identity, optimizer_G,
      optimizer_D_R, optimizer_D_M, lr_scheduler_G, lr_scheduler_D_R, lr_scheduler_D_M, epoch_num)
