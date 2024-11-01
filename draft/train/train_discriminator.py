import torch
import torch.nn as nn
import torch.optim as optim
from numpy.ma.core import shape
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
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


class Discriminator(nn.Module):
    def __init__(self,num_residual_blocks=9):
        super(Discriminator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks // 2)],
            SelfAttention(64),
            *[ResidualBlock(64) for _ in range(num_residual_blocks // 2)]
        )
        self.output=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Ensure output is between 0 and 1
        )



    def forward(self, x):
        x = self.initial(x)
        x = self.residual_blocks(x)
        x = self.output(x)
        return x

train_preprocess = transforms.Compose([
    transforms.Resize((286, 286)),  # 放大
    transforms.RandomCrop((256, 256)),  # 随机裁剪为256x256
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 颜色抖动
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])


def image_loader(path):
    return Image.open(path).convert('RGB')

data_base_train = datasets.ImageFolder('../pics/train', transform=train_preprocess)
data_base_val = datasets.ImageFolder('../pics/train', transform=train_preprocess)

batch_size = 30

train_loader = DataLoader(data_base_train, batch_size=batch_size, shuffle=True,drop_last=True)
val_loader = DataLoader(data_base_val, batch_size=batch_size, shuffle=True,drop_last=True)

D_divide = Discriminator().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(D_divide.parameters(), lr=0.0002, betas=(0.5, 0.999))

def train(num_epochs=20):
    for epoch in range(num_epochs):
        D_divide.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,1,256,256)
            optimizer.zero_grad()
            outputs = D_divide(inputs)
            print(f"i = {i}\nlabels:{labels.size()}\noutputs:{outputs.size()}\n")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/50:.4f}')
                running_loss = 0.0
        D_divide.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,1,256,256)
                outputs = D_divide(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        if epoch%10 == 1:
            torch.save(D_divide.state_dict(), f'../model/Dis/D_divide_{epoch}_{val_loss/len(val_loader):.4f}loss.pth')

        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss/len(val_loader):.4f}')


train()