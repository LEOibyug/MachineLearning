import torch
import torch.nn as nn
import torch.optim as optim
from functorch.einops import rearrange
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


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = torch.nn.functional.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ffn_expansion_factor=2.66):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim, num_heads, bias=False)
        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias=False)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            TransformerBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            TransformerBlock(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 3, kernel_size=3, stride=1, padding=1),
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
D.load_state_dict(torch.load('../../models/Dis/25_Dis.pth', map_location=device, weights_only=True))
D.to(device)
D.eval()
real_image_path = '../../pics/mixed'
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
    axs[pic].set_title(f'P = R:{score[0][0][0]:.4f} G:{score[1][0][0]:.4f} B:{score[2][0][0]:.4f} A = {answer}')
    axs[pic].axis('off')


plt.show()