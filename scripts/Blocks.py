import torch
import torch.nn as nn


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

class ResidualBlock_o1(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock_o1, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),  # 使用ReflectionPad2d代替padding，可以减少边缘效应
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1),
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


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super(MultiHeadSelfAttention, self).__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()

        query = self.query_conv(x).view(batch_size, self.num_heads, self.head_dim, width * height)
        key = self.key_conv(x).view(batch_size, self.num_heads, self.head_dim, width * height)
        value = self.value_conv(x).view(batch_size, self.num_heads, self.head_dim, width * height)

        query = query.permute(0, 1, 3, 2)
        key = key.permute(0, 1, 2, 3)

        attention = torch.matmul(query, key) / (self.head_dim ** 0.5)
        attention = torch.softmax(attention, dim=-1)

        value = value.permute(0, 1, 3, 2)
        out = torch.matmul(attention, value)

        out = out.permute(0, 1, 3, 2).contiguous().view(batch_size, C, width * height)
        out = out.view(batch_size, C, width, height)

        out = self.gamma * out + x
        return out


class SelfAttention_o1(nn.Module):
    """Self-Attention Layer as per Zhang et al., 2018."""
    def __init__(self, in_dim):
        super(SelfAttention_o1, self).__init__()
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

class ResidualBlockAtt(nn.Module):
    def __init__(self, in_channels, use_attention=False):
        super(ResidualBlockAtt,self).__init__()
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
            self.attention = SelfAttention_o1(in_channels)

    def forward(self, x):
        out = self.conv_block(x)
        if self.use_attention:
            out = self.attention(out)
        return x + out
