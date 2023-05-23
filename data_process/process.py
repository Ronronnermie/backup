import h5py
import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# 四种模态的mri图像
modalities = ('flair', 't1ce', 't1', 't2')

# train
train_set = {
    'root': '../../BraTS2021/archive/data',  # 四个模态数据所在地址
    'out': '../../BraTS2021/archive/data/',  # 预处理输出地址
    'flist': 'train.txt',  # 训练集名单（有标签）
}


def process_h5(path, out_path):
    """ Save the data with dtype=float32.
        z-score is used but keep the background with zero! """
    # SimpleITK读取图像默认是是 DxHxD，这里转为 HxWxD
    label = sitk.GetArrayFromImage(sitk.ReadImage(path + 'seg.nii.gz')).transpose(1, 2, 0)
    print(label.shape)
    # 堆叠四种模态的图像，4 x (H,W,D) -> (4,H,W,D)
    images = np.stack(
        [sitk.GetArrayFromImage(sitk.ReadImage(path + modal + '.nii.gz')).transpose(1, 2, 0) for modal in modalities],
        0)  # [240,240,155]
    # 数据类型转换
    label = label.astype(np.uint8)
    images = images.astype(np.float32)
    case_name = path.split('/')[-1]
    # case_name = os.path.split(path)[-1]  # windows路径与linux不同

    path = os.path.join(out_path, case_name)
    output = path + 'mri_norm2.h5'
    # 对第一个通道求和，如果四个模态都为0，则标记为背景(False)
    mask = images.sum(0) > 0
    for k in range(4):
        x = images[k, ...]  #
        y = x[mask]

        # 对背景外的区域进行归一化
        x[mask] -= y.mean()
        x[mask] /= y.std()

        images[k, ...] = x
    print(case_name, images.shape, label.shape)
    f = h5py.File(output, 'w')
    f.create_dataset('image', data=images, compression="gzip")
    f.create_dataset('label', data=label, compression="gzip")
    f.close()


def doit(dset):
    root, out_path = dset['root'], dset['out']
    file_list = os.path.join(root, dset['flist'])
    subjects = open(file_list).read().splitlines()
    names = ['BraTS2021_' + sub for sub in subjects]
    paths = [os.path.join(root, name, name + '_') for name in names]

    for path in tqdm(paths):
        process_h5(path, out_path)
        # break
    print('Finished')


if __name__ == '__main__':
    doit(train_set)

# import nibabel as nib
# import numpy as np
# import os
# import imageio
# import matplotlib
# matplotlib.use('TkAgg')

# ---------------------------------------------#
# nii_path : nii文件的路径
# img_save_path : 切片的保存路径
# axis : 说明是沿着哪个方向切片的
# ---------------------------------------------


# def nii_to_png(nii_path, img_save_path, axis):  #将NIfTI格式的3D脑部图像文件转换为PNG格式的二维图像文件，并将其保存到指定路径下。
#     # 若保存路径不存在，则创建
#     global slice
#     if not os.path.exists(img_save_path):
#         os.makedirs(img_save_path)
#
#     nii = nib.load(nii_path)   #读取指定路径下的NIfTI格式图像文件
#     nii_fdata = nii.get_fdata()  #图像数据转换为NumPy数组。
#     # nii_fdata = np.rot90(nii_fdata)
#
#     # 以切片的轴向作为保存png的子文件夹名
#     foldername = axis
#     png_save_path = os.path.join(img_save_path, foldername)
#     if not os.path.exists(png_save_path):
#         os.mkdir(png_save_path)
#    #根据输入的切片轴向axis，将3D图像数据沿着指定轴向进行切片，并将每个切片保存为一个PNG格式的二维图像文件。
#    # PNG图像文件的命名规则为数字1、2、3…，保存在以切片轴向命名的子文件夹中。
#     flag = 100
#     if axis == 'x':
#         (axis, y, z) = nii.shape
#         flag = 0
#     elif axis == 'y':
#         (x, axis, z) = nii.shape
#         flag = 1
#     elif axis == 'z':
#         (x, y, axis) = nii.shape
#         flag = 2
#     else:
#         print("wrong axis")
#
#     for i in range(axis):
#         if flag == 0:
#             slice = nii_fdata[i, :, :]
#         elif flag == 1:
#             slice = nii_fdata[:, i, :]
#         elif flag == 2:
#             slice = nii_fdata[:, :, i]
#         # 以数字1,2,3...为png图片命名
#         imageio.imwrite(os.path.join(png_save_path, '{}.png'.format(i)), slice)
#
#
# def all_nii_to_png(all_nii_path,all_image_save_path, axis):
#     all_nii_path_list = os.listdir(all_nii_path)
#     for i in range(len(all_nii_path_list)):
#         nii_to_png(os.path.join(all_nii_path, all_nii_path_list[i]), os.path.join(all_image_save_path, all_nii_path_list[i]), axis)
#         print("第{}个nii文件转换完成".format(i))
#
#
# if __name__ == "__main__":
#     all_nii_path = "../../BraTS2021/archive/data/BraTS2021_00000"
#     all_image_save_path = "../../BraTS2021/archive/dataset"
#     all_nii_to_png(all_nii_path, all_image_save_path, 'z')
