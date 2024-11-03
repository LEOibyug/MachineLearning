import random
import matplotlib.pyplot as plt
import os
import math
from scripts.Networks import *
from scripts.directions import *
import re
from torchvision import transforms
from PIL import Image
import time
########################################################################################################
model_name = 'GeneratorAttO1_312.pth'
########################################################################################################
pattern = r'^[^_]+'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
match = re.match(pattern, model_name)
modelType = match.group(0)
print(f"modelType: {modelType}")
mode_dir = G_R2M_SAVE + model_name
exec(f'G_R2M = {modelType}().to(device)')
G_R2M.load_state_dict(torch.load(mode_dir, weights_only=True,map_location=device))
G_R2M.eval()

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

real_image_path = REAL  # 随机读取目录
real_images = os.listdir(real_image_path)
########################################################################################################
pic_num = 10
# 是否随机生成
random_draw = True
pic_d = ['./11.png']  # 指定图片
########################################################################################################
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
    plt.show()
    fig.savefig(PIC_SAVE+f'{int(time.mktime(time.localtime()))}.png', dpi=300)

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
    plt.show()
    fig.savefig(PIC_SAVE+f'{int(time.mktime(time.localtime()))}.png', dpi=300)


else:
    print('No images to display')

