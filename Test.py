import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
import itertools
from PIL import Image
import os
from scripts.directions import *

# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 设置可用的GPU设备ID列表，例如使用0号和1号GPU
device_ids = [0, 1]

# 定义自注意力模块
class SelfAttention(nn.Module):
    """Self-Attention Layer as per Zhang et al., 2018."""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.in_channel = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 学习参数gamma

    def forward(self, x):
        batch_size, C, width, height = x.size()

        # Compute query, key and value matrices
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B x N x C'
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)  # B x C' x N
        energy = torch.bmm(proj_query, proj_key)  # B x N x N
        attention = self.softmax(energy)  # B x N x N

        proj_value = self.value_conv(x).view(batch_size, -1, width * height)  # B x C x N

        # Compute attention output
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x N
        out = out.view(batch_size, C, width, height)

        # Apply the scaling parameter gamma
        out = self.gamma * out + x
        return out

# 定义残差块，加入自注意力机制（可选）
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, use_attention=False):
        super(ResidualBlock, self).__init__()
        self.use_attention = use_attention
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),  # 使用ReflectionPad2d代替padding，可以减少边缘效应
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(in_channels)
        )
        if use_attention:
            self.attention = SelfAttention(in_channels)

    def forward(self, x):
        out = self.conv_block(x)
        if self.use_attention:
            out = self.attention(out)
        return x + out

# 定义生成器，加入自注意力机制
class Generator(nn.Module):
    def __init__(self, num_residual_blocks=9):
        super(Generator, self).__init__()
        # Initial convolution block
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7),
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
        # Residual blocks with self-attention
        res_blocks = []
        for i in range(num_residual_blocks):
            # 在第5个残差块后加入自注意力机制
            if i == num_residual_blocks // 2:
                res_blocks.append(ResidualBlock(256, use_attention=True))
            else:
                res_blocks.append(ResidualBlock(256))
        self.residual_blocks = nn.Sequential(*res_blocks)

        # Upsampling
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Output layer
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.downsampling(x)
        x = self.residual_blocks(x)
        x = self.upsampling(x)
        return self.output(x)

# 定义判别器，加入自注意力机制
class Discriminator(nn.Module):
    def __init__(self, input_nc=3):
        super(Discriminator, self).__init__()
        # 使用70x70 PatchGAN
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=4,
                      stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4,
                      stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4,
                      stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 在第三层后加入自注意力机制
        self.attention = SelfAttention(256)
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4,
                      stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.output_layer = nn.Conv2d(512, 1, kernel_size=4,
                                      stride=1, padding=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.attention(x)  # 自注意力层
        x = self.layer4(x)
        return self.output_layer(x)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 自定义数据集
def image_loader(path):
    return Image.open(path).convert('RGB')

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_real, root_monet, transform=None):
        self.transform = transform
        self.real_images = sorted([os.path.join(root_real, img) for img in os.listdir(root_real)])
        self.monet_images = sorted([os.path.join(root_monet, img) for img in os.listdir(root_monet)])
        self.length = max(len(self.real_images), len(self.monet_images))

    def __getitem__(self, index):
        real_img = image_loader(self.real_images[index % len(self.real_images)])
        monet_img = image_loader(self.monet_images[index % len(self.monet_images)])
        if self.transform:
            real_img = self.transform(real_img)
            monet_img = self.transform(monet_img)
        return {'real': real_img, 'monet': monet_img}

    def __len__(self):
        return self.length

# 创建数据加载器
batch_size = 2  # 调整批量大小为2，以适应双卡训练
dataset = ImageDataset(T_REAL + 'c1/', T_MONET + 'c1/', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# 初始化网络
G_R2M = Generator()
G_M2R = Generator()
D_R = Discriminator()
D_M = Discriminator()

# 使用DataParallel包装模型
G_R2M = nn.DataParallel(G_R2M, device_ids=device_ids).to(device)
G_M2R = nn.DataParallel(G_M2R, device_ids=device_ids).to(device)
D_R = nn.DataParallel(D_R, device_ids=device_ids).to(device)
D_M = nn.DataParallel(D_M, device_ids=device_ids).to(device)

# 定义损失函数
criterion_GAN = nn.MSELoss().to(device)
criterion_cycle = nn.L1Loss().to(device)
criterion_identity = nn.L1Loss().to(device)

# 定义优化器
lr = 0.0002
optimizer_G = optim.Adam(itertools.chain(G_R2M.parameters(), G_M2R.parameters()), lr=lr, betas=(0.5, 0.999))
optimizer_D_R = optim.Adam(D_R.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_M = optim.Adam(D_M.parameters(), lr=lr, betas=(0.5, 0.999))

# 学习率调度器
lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 300) / 100)
lr_scheduler_D_R = optim.lr_scheduler.LambdaLR(optimizer_D_R, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 300) / 100)
lr_scheduler_D_M = optim.lr_scheduler.LambdaLR(optimizer_D_M, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 300) / 100)

# 定义一个缓存，用于存储生成的假样本，提升判别器的稳定性
class ReplayBuffer():
    def __init__(self, max_size=50):
        assert max_size > 0, "缓冲区大小应大于0"
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.detach().cpu().data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if torch.rand(1).item() > 0.5:
                    idx = torch.randint(0, len(self.data), (1,)).item()
                    tmp = self.data[idx].clone()
                    self.data[idx] = element
                    to_return.append(tmp)
                else:
                    to_return.append(element)
        return torch.cat(to_return).to(device)

fake_M_buffer = ReplayBuffer()
fake_R_buffer = ReplayBuffer()

# 开始训练
epoch_num = 400
for epoch in range(1, epoch_num + 1):
    for i, batch in enumerate(dataloader):
        # 获取数据
        real_R = batch['real'].to(device)
        real_M = batch['monet'].to(device)

        # ------------------
        #  训练生成器
        # ------------------
        optimizer_G.zero_grad()

        # Identity loss
        loss_id_R = criterion_identity(G_M2R(real_R), real_R) * 5.0
        loss_id_M = criterion_identity(G_R2M(real_M), real_M) * 5.0

        # GAN loss
        fake_M = G_R2M(real_R)
        pred_fake_M = D_M(fake_M)
        valid = torch.ones_like(pred_fake_M, requires_grad=False)
        loss_GAN_R2M = criterion_GAN(pred_fake_M, valid)

        fake_R = G_M2R(real_M)
        pred_fake_R = D_R(fake_R)
        valid = torch.ones_like(pred_fake_R, requires_grad=False)
        loss_GAN_M2R = criterion_GAN(pred_fake_R, valid)

        # Cycle loss
        recov_R = G_M2R(fake_M)
        loss_cycle_R = criterion_cycle(recov_R, real_R) * 10.0

        recov_M = G_R2M(fake_R)
        loss_cycle_M = criterion_cycle(recov_M, real_M) * 10.0

        # 总的生成器损失
        loss_G = loss_GAN_R2M + loss_GAN_M2R + loss_cycle_R + loss_cycle_M + loss_id_R + loss_id_M

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  训练判别器 D_R
        # -----------------------
        optimizer_D_R.zero_grad()

        # 使用历史生成的假的图片来训练判别器，提高稳定性
        fake_R_ = fake_R_buffer.push_and_pop(fake_R)

        pred_real_R = D_R(real_R)
        valid = torch.ones_like(pred_real_R, requires_grad=False)
        loss_D_R_real = criterion_GAN(pred_real_R, valid)

        pred_fake_R_ = D_R(fake_R_.detach())
        fake = torch.zeros_like(pred_fake_R_, requires_grad=False)
        loss_D_R_fake = criterion_GAN(pred_fake_R_, fake)

        loss_D_R = (loss_D_R_real + loss_D_R_fake) * 0.5

        loss_D_R.backward()
        optimizer_D_R.step()

        # -----------------------
        #  训练判别器 D_M
        # -----------------------
        optimizer_D_M.zero_grad()

        fake_M_ = fake_M_buffer.push_and_pop(fake_M)

        pred_real_M = D_M(real_M)
        valid = torch.ones_like(pred_real_M, requires_grad=False)
        loss_D_M_real = criterion_GAN(pred_real_M, valid)

        pred_fake_M_ = D_M(fake_M_.detach())
        fake = torch.zeros_like(pred_fake_M_, requires_grad=False)
        loss_D_M_fake = criterion_GAN(pred_fake_M_, fake)

        loss_D_M = (loss_D_M_real + loss_D_M_fake) * 0.5

        loss_D_M.backward()
        optimizer_D_M.step()

        # --------------
        #  输出日志信息
        # --------------
       
    print(f"[Epoch {epoch}/{epoch_num}] [Batch {i}/{len(dataloader)}] "
                  f"[D_R loss: {loss_D_R.item():.4f}] [D_M loss: {loss_D_M.item():.4f}] "
                  f"[G loss: {loss_G.item():.4f}]")
    # 更新学习率
    lr_scheduler_G.step()
    lr_scheduler_D_R.step()
    lr_scheduler_D_M.step()

    # 保存模型
    if epoch % 2 == 0:
        # 为了兼容DataParallel，需要调整保存方式
        torch.save(G_R2M.module.state_dict(), f'{G_R2M_SAVE}GeneratorAtt_o1_{epoch}.pth')
        torch.save(G_M2R.module.state_dict(), f'{G_R2M_SAVE}GeneratorAtt_o1_anti_{epoch}.pth')

print('Training finished.')