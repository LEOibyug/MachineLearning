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
import random


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

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return preprocess(image).unsqueeze(0).to(device)

def infer(image_path, model):
    image = load_image(image_path)
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda'):
            fake_image = model(image)
    return fake_image.squeeze().cpu().detach().numpy()



image_dir = ''
D = Discriminator()
D.load_state_dict(torch.load('../modules/Dis/D_divide_28_0.1695loss.pth',map_location=device,weights_only=True))
D.to(device)
D = D.half()
D.eval()
real_image_path = '../pics/mixed'
real_images = os.listdir(real_image_path)
pic_num = 10
# 显示图像
fig, axs = plt.subplots(pic_num, 1, figsize=(8, 3 * pic_num))
for pic in range(pic_num):
    pic_id = random.randint(0, len(real_images) - 1)
    real_image_name = real_image_path + '/' + real_images[pic_id]
    print(real_image_name)
    score = infer(real_image_name, D)

    # 原始图像
    axs[pic].imshow(Image.open(real_image_name))
    if len(real_images[pic_id]) > 9:
        answer = 'Real'
    else:
        answer = 'Monet'
    axs[pic].set_title(f'P = {1-score.mean():.4f} discriminate_T1.pyA = {answer}')
    axs[pic].axis('off')


plt.show()