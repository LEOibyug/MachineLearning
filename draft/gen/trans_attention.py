import random
import matplotlib.pyplot as plt
from PIL import Image
import os
import math
import torch
import torch.nn as nn
from torchvision import transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)


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



# 加载训练好的模型
G_R2M = Generator().to(device)
G_R2M.load_state_dict(torch.load('./results/G_R2M_epoch_626_enlarge.pth', map_location=device, weights_only=True))
G_R2M.eval()


# 图像预处理

class DynamicResizeTransform:
    def __call__(self, img):
        width, height = img.size
        # 根据尺寸进行动态缩放,防止爆显存
        if width * height > 600000:
            scale_factor = math.sqrt(600000 / (width * height))
        else:
            scale_factor = 1.0
        new_width = round(width * scale_factor)
        new_height = round(height * scale_factor)
        print(f'origin {height}*{width}  output {new_height}*{new_width}')
        return img.resize((new_width, new_height))


dynamic_resize = DynamicResizeTransform()

transform = transforms.Compose([
    dynamic_resize,
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# 加载图像
def load_image(img_path):
    image = Image.open(img_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device)


# 图像转换
def convert_image(image_tensor):
    with torch.no_grad():
        fake_image = G_R2M(image_tensor)
    return fake_image


# 可视化图像
def show_image(tensor):
    image = tensor.cpu().clone().detach().squeeze(0)
    image = image * 0.5 + 0.5
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


real_image_path = './photo_jpg'  # 随机读取目录
real_images = os.listdir(real_image_path)
pic_num = 10
# 是否随机生成
random_draw = True

pic_d = ['./temp_pic/3333.png']  # 指定图片
if random_draw:
    fig, axs = plt.subplots(pic_num, 2, figsize=(8, 3 * pic_num))
    for pic in range(pic_num):
        pic_id = random.randint(0, len(real_images) - 1)
        real_image_name = os.path.join(real_image_path, real_images[pic_id])
        real_image_tensor = load_image(real_image_name)
        fake_monet_image_tensor = convert_image(real_image_tensor)

        if pic_num > 1:
            axs[pic, 0].imshow(Image.open(real_image_name))
            axs[pic, 0].set_title('Original Real Image')
            axs[pic, 0].axis('off')

            fake_monet_image = fake_monet_image_tensor.cpu().clone().detach().squeeze(0)
            fake_monet_image = fake_monet_image * 0.5 + 0.5  # 反归一化
            fake_monet_image = transforms.ToPILImage()(fake_monet_image)
            axs[pic, 1].imshow(fake_monet_image)
            axs[pic, 1].set_title('Fake Monet Image')
            axs[pic, 1].axis('off')
        else:
            axs[0].imshow(Image.open(real_image_name))
            axs[0].set_title('Original Real Image')
            axs[0].axis('off')

            fake_monet_image = fake_monet_image_tensor.cpu().clone().detach().squeeze(0)
            fake_monet_image = fake_monet_image * 0.5 + 0.5  # 反归一化
            fake_monet_image = transforms.ToPILImage()(fake_monet_image)
            axs[1].imshow(fake_monet_image)
            axs[1].set_title('Fake Monet Image')
            axs[1].axis('off')
elif len(pic_d) > 0:
    pic_num = len(pic_d)
    fig, axs = plt.subplots(pic_num, 2, figsize=(8, 3 * pic_num))
    for pic in range(pic_num):
        pic_id = random.randint(0, len(real_images) - 1)
        real_image_name = pic_d[pic]
        real_image_tensor = load_image(real_image_name)
        fake_monet_image_tensor = convert_image(real_image_tensor)
        if pic_num > 1:
            axs[pic, 0].imshow(Image.open(real_image_name))
            axs[pic, 0].set_title('Original Real Image')
            axs[pic, 0].axis('off')

            fake_monet_image = fake_monet_image_tensor.cpu().clone().detach().squeeze(0)
            fake_monet_image = fake_monet_image * 0.5 + 0.5  # 反归一化
            fake_monet_image = transforms.ToPILImage()(fake_monet_image)
            axs[pic, 1].imshow(fake_monet_image)
            axs[pic, 1].set_title('Fake Monet Image')
            axs[pic, 1].axis('off')
        else:
            axs[0].imshow(Image.open(real_image_name))
            axs[0].set_title('Original Real Image')
            axs[0].axis('off')

            fake_monet_image = fake_monet_image_tensor.cpu().clone().detach().squeeze(0)
            fake_monet_image = fake_monet_image * 0.5 + 0.5  # 反归一化
            fake_monet_image = transforms.ToPILImage()(fake_monet_image)
            axs[1].imshow(fake_monet_image)
            axs[1].set_title('Fake Monet Image')
            axs[1].axis('off')
else:
    print('No images to display')
plt.show()
