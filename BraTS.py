import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from torchvision.transforms import transforms
import h5py


# 随机裁剪
class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        (c, w, h, d) = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[:,w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}


# 中心裁剪
class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        (c, w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[:,w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


# 随机翻转
class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.stack([np.rot90(x, k) for x in image], axis=0)
        label = np.rot90(label, k)
        axis = np.random.randint(1, 4)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis-1).copy()

        return {'image': image, 'label': label}


# 高斯噪声
def augment_gaussian_noise(data_sample, noise_variance=(0, 0.1)):
    if noise_variance[0] == noise_variance[1]:
        variance = noise_variance[0]
    else:
        variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
    return data_sample


class GaussianNoise(object):
    def __init__(self, noise_variance=(0, 0.1), p=0.5):
        self.prob = p
        self.noise_variance = noise_variance

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if np.random.uniform() < self.prob:
            image = augment_gaussian_noise(image, self.noise_variance)
        return {'image': image, 'label': label}


# 数据类型转换 numpy转为tensor
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return {'image': image, 'label': label}




class BraTS(Dataset):
    def __init__(self, data_path, file_path, transform=None):
        with open(file_path, 'r') as f: #file_path要打开的文件路径，‘r’打开文件的模式：只读，打开文件对象赋值给f
            self.paths = [os.path.join(data_path, x.strip()) for x in f.readlines()]    #形成了 MRI 扫描文件的完整路径,被保存在 self.paths 列表
        self.transform = transform     #将data_path和文件中的每一行（去掉首尾空格）拼接起来，得到一个完整的文件路径。

    def __getitem__(self, item):
        h5f = h5py.File(self.paths[item], 'r')     # 用 h5py.File 函数打开了一个 MRI 扫描文件，从中读取image和label数据
        image = h5f['image'][:]  #[:]表示读取整个image内容
        label = h5f['label'][:]
        #[0,1,2,4] -> [0,1,2,3]
        label[label == 4] = 3
        # print(image.shape)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample['image'], sample['label']

    def __len__(self):
        return len(self.paths)

    def collate(self, batch):    #将多个样本合并为一个batch
        return [torch.cat(v) for v in zip(*batch)]


if __name__ == '__main__':
    from torchvision import transforms
    data_path = "../BraTS2021/archive/dataset"
    test_txt = "../BraTS2021/archive/test.txt"
    test_set = BraTS(data_path, test_txt, transform=transforms.Compose([
        RandomRotFlip(),
        RandomCrop((160, 160, 128)),
        GaussianNoise(p=0.1),
        ToTensor()
    ]))
    d1 = test_set[0]  #从BraTS数据集对象中读取第一个测试数据。
    image, label = d1  #将测试数据d1中的图像数据和标签数据分别赋值给变量image和label。这里假设d1是一个元组，其中第一个元素为图像数据，第二个元素为标签数据。
    print(image.shape)
    print(label.shape)
    print(np.unique(label))
