import random
import torch
import torch.nn as nn
from torchvision import transforms
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


class Generator(nn.Module):
    def __init__(self, num_residual_blocks=9):
        super(Generator, self).__init__()
        # Initial convolution block
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Downsampling
        self.downsampling = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_residual_blocks)]
        )

        # Upsampling
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Output layer
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


# 定义图像处理步骤
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return preprocess(image).unsqueeze(0).to(device)


def denormalize(tensor):
    tensor = tensor * 0.5 + 0.5  # 反归一化
    return tensor.clamp(0, 1)


# 加载模型
G_R2M = Generator().to(device)
# G_R2M = torch.load("./archive/data.pkl")
G_R2M.load_state_dict(torch.load('./results/G_R2M_epoch_626_enlarge.pth', weights_only=True))
G_R2M.eval()


def infer(image_path, model):
    image = load_image(image_path)
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda'):
            fake_image = model(image)
    fake_image = denormalize(fake_image.squeeze(0).cpu())
    return transforms.ToPILImage()(fake_image)


# 推理
real_image_path = './photo_jpg'
real_images = os.listdir(real_image_path)
pic_num = 10
# 显示图像
fig, axs = plt.subplots(pic_num, 2, figsize=(8, 3 * pic_num))

for pic in range(pic_num):
    pic_id = random.randint(0, len(real_images) - 1)
    real_image_name = './photo_jpg/' + real_images[pic_id]
    print(real_image_name)
    fake_monet_image = infer(real_image_name, G_R2M)

    # 原始图像
    axs[pic, 0].imshow(Image.open(real_image_name))
    axs[pic, 0].set_title('Original Real Image')
    axs[pic, 0].axis('off')

    # 转换后的图像
    axs[pic, 1].imshow(fake_monet_image)
    axs[pic, 1].set_title('Fake Monet Image')
    axs[pic, 1].axis('off')

plt.show()
