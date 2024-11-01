from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from PIL import Image


train_preprocess = transforms.Compose([
    transforms.Resize((286, 286)),  # 首先将图像放大
    transforms.RandomCrop((256, 256)),  # 然后随机裁剪为256x256
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 颜色抖动
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])


def image_loader(path):
    return Image.open(path).convert('RGB')

def get_data_loader(dir,batchSize):
    data_base = DatasetFolder(dir, loader=image_loader,extensions=('jpg', 'jpeg', 'png'),transform=train_preprocess)
    dataloader = DataLoader(data_base, batch_size=batchSize, shuffle=True)
    return dataloader


