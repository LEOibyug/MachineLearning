import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
import cv2
from functorch.einops import rearrange

device = 'cuda'
max_epoch = 100
batch_size = 8


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

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

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
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

train_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((286, 286)),  # 放大
    transforms.RandomCrop((256, 256)),  # 随机裁剪为256x256
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 颜色抖动
    # AddGaussianNoise(mean=0.0, std=0.1),  # 随机加入一些高斯噪声
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])


class Data(Dataset):
    def __init__(self, folder_path, transform, label):
        super().__init__()
        imgs, labels = [], []
        for img_path in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, img_path))
            img = transform(img)
            imgs.append(img)
            if label == 1:
                labels.append(torch.ones_like(img))
            else:
                labels.append(torch.zeros_like(img))
        self.imgs = torch.stack(imgs)
        self.labels = torch.stack(labels)

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]
    
monet_dataset = Data('../../pics/train/Monet', train_preprocess, 1)
real_dataset = Data('../../pics/train/Real', train_preprocess, 0)
dataset = ConcatDataset([monet_dataset, real_dataset])
dataloader = DataLoader(dataset, batch_size, shuffle=True)

model = Discriminator().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
criterion = nn.MSELoss()
scaler = GradScaler()

for epoch in range(max_epoch):
    for step, (img, label) in enumerate(dataloader):
        img = img.to(device)
        label = label.to(device)
        # pred = model(img)
        # loss = criterion(pred, label)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        
        with autocast():
            pred = model(img)
            loss = criterion(pred, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print(f'Epoch:{epoch+1}, Step:{step+1}, Loss:{loss.item()}')
    torch.save(model.state_dict(), f'../modules/{epoch+1}_Dis.pth')
