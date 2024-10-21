from matplotlib import pyplot as plt
import numpy as np
import re

with open('./results/nohup.out', 'r', encoding='utf-8') as file:
    datas = file.readlines()

# 正则表达式模式，用于匹配整数和小数
pattern = r'\d+(\.\d+)?'
datas.pop(0)
data = np.array([d.split(',') for d in datas])
loss = data[:, 2]
ys = []
for text in loss:
    match = re.search(pattern, text)
    if match:
        # 提取匹配的数字
        number = match.group()
        ys.append(float(number))
index = np.argmin(ys)
print(data[index])
xs = range(1, len(ys) + 1)
plt.plot(xs, ys)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()