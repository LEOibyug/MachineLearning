from django.shortcuts import render
from pathlib import Path

def index(request):
    # 图片文件夹路径
    image_folder = Path('/home/u2023112931/MachineLearning/pics/outputs')
    
    # 获取所有图片文件，并按文件名中的时间戳排序
    images = sorted(
        [f for f in image_folder.iterdir() if f.is_file() and f.suffix in ['.jpg', '.jpeg', '.png']],
        key=lambda x: int(x.stem),
        reverse=True
    )
    
    # 取前9张图片或所有图片
    images = images[:9] if images else None
    
    # 将图片路径传递给模板
    if images:
        context = {'images': [f'/media/{image.name}' for image in images]}
    else:
        context = {'message': '没有可展示的图片。'}
    
    return render(request, 'gallery/index.html', context)
