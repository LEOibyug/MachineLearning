import os
import re
# 正则表达式
pattern = r".*?MachineLearning"

# 使用re.match来获取匹配的部分
Root_path = os.path.abspath(__file__).replace('\\', '/')
Root_path = re.match(pattern, Root_path).group(0)
T_MONET = Root_path + '/pics/train/Monet/'
T_REAL = Root_path + '/pics/train/Real/'
MIX = Root_path + '/pics/mixed/'
REAL = Root_path + '/pics/archives/Real/'
TEMP = Root_path + '/pics/temp/'
G_R2M_SAVE = Root_path + '/models/G_R2M/'
PIC_SAVE = Root_path + '/pics/output/'