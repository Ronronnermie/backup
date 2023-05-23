import os
from sklearn.model_selection import train_test_split

# 预处理输出地址
data_path = "../../BraTS2021/archive/dataset"
train_and_test_ids = os.listdir(data_path)   #返回指定路径下的文件和文件夹列表。

# 将数据集按照 8:1:1随机划分为训练集、验证集和测试集，将划分后的数据名保存为.txt文件
train_ids, val_test_ids = train_test_split(train_and_test_ids, test_size=0.2,random_state=21)  #1验证测试集占20%
val_ids, test_ids = train_test_split(val_test_ids, test_size=0.5,random_state=21) #验证集、测试集各占1的50%
print("Using {} images for training, {} images for validation, {} images for testing.".format(len(train_ids),len(val_ids),len(test_ids)))

# 数组排序？
train_ids.sort()
val_ids.sort()
test_ids.sort()

with open('../../BraTS2021/archive/train.txt', 'w') as f:
    f.write('\n'.join(train_ids))

with open('../../BraTS2021/archive/valid.txt', 'w') as f:
    f.write('\n'.join(val_ids))

with open('../../BraTS2021/archive/test.txt', 'w') as f:
    f.write('\n'.join(test_ids))
