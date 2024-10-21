import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import itertools
import matplotlib.pyplot as plt
from PIL import Image
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'using device {device}')

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
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
        x = self.model(x)
        return x

# class AddGaussianNoise(object):
#     def __init__(self, mean=0.0, std=1.0):
#         self.mean = mean
#         self.std = std
#
#     def __call__(self, tensor):
#         return tensor + torch.randn(tensor.size()) * self.std + self.mean
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

train_preprocess = transforms.Compose([
    transforms.Resize((286, 286)),  # 放大
    transforms.RandomCrop((256, 256)),  # 随机裁剪为256x256
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 颜色抖动
    # AddGaussianNoise(mean=0.0, std=0.1),  # 随机加入一些高斯噪声
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])


def image_loader(path):
    return Image.open(path).convert('RGB')

data_base_train = datasets.ImageFolder('../pics/train', transform=train_preprocess)
data_base_val = datasets.ImageFolder('../pics/test', transform=train_preprocess)

batch_size = 3

train_loader = DataLoader(data_base_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(data_base_val, batch_size=batch_size, shuffle=True)

D_divide = Discriminator().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(D_divide.parameters(), lr=0.0002, betas=(0.5, 0.999))

def train(num_epochs=20):
    for epoch in range(num_epochs):
        D_divide.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device).float().view(-1, 1,1,1).repeat(1,1,256,256)
            optimizer.zero_grad()
            outputs = D_divide(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 50 == 49:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/50:.4f}')
                running_loss = 0.0

        D_divide.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float().view(-1, 1)
                outputs = D_divide(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss/len(val_loader):.4f}')


train()