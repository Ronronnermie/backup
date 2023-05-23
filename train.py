import os
import argparse

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
from BraTS import *
from networks.unet import UNet
from utils import Loss,cal_dice,cosine_scheduler


def train_loop(model, optimizer, scheduler, criterion, train_loader, device, epoch):
    model.train()  #模型设为训练模式                         #损失函数   训练数据集的数据加载器
    running_loss = 0
    dice1_train = 0
    dice2_train = 0
    dice3_train = 0
    pbar = tqdm(train_loader)  #创建一个进度条，用于显示训练进度
    for it, (images, masks) in enumerate(pbar):  #遍历训练数据集
        # update learning rate according to the schedule
        it = len(train_loader) * epoch + it   #计算当前迭代次数
        param_group = optimizer.param_groups[0]  #获取优化器的参数组
        param_group['lr'] = scheduler[it]   #更新学习率
        # print(scheduler[it])

        # [b,4,128,128,128] , [b,128,128,128]
        images, masks = images.to(device), masks.to(device) #数据和标签移到指定设备上
        # [b,4,128,128,128], 4分割
        outputs = model(images)     #前向传播，得到模型输出
        # outputs = torch.softmax(outputs,dim=1)
        loss = criterion(outputs, masks)     #计算损失值
        dice1, dice2, dice3 = cal_dice(outputs, masks)   #计算三个类别的dice系数
        pbar.desc = "loss: {:.3f} ".format(loss.item())   #更新进度条描述
        # 累加
        running_loss += loss.item()
        dice1_train += dice1.item()
        dice2_train += dice2.item()
        dice3_train += dice3.item()
        # 清空梯度
        optimizer.zero_grad()
        loss.backward()  #反向传播
        optimizer.step() #更新模型参数
    # 计算平均
    loss = running_loss / len(train_loader)
    dice1 = dice1_train / len(train_loader)
    dice2 = dice2_train / len(train_loader)
    dice3 = dice3_train / len(train_loader)
    return {'loss': loss, 'dice1': dice1, 'dice2': dice2, 'dice3': dice3}


def val_loop(model, criterion, val_loader, device):
    model.eval()
    running_loss = 0
    dice1_val = 0
    dice2_val = 0
    dice3_val = 0
    pbar = tqdm(val_loader)
    with torch.no_grad():
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            # outputs = torch.softmax(outputs,dim=1)

            loss = criterion(outputs, masks)
            dice1, dice2, dice3 = cal_dice(outputs, masks)

            running_loss += loss.item()
            dice1_val += dice1.item()
            dice2_val += dice2.item()
            dice3_val += dice3.item()
            # pbar.desc = "loss:{:.3f} dice1:{:.3f} dice2:{:.3f} dice3:{:.3f} ".format(loss,dice1,dice2,dice3)

    loss = running_loss / len(val_loader)
    dice1 = dice1_val / len(val_loader)
    dice2 = dice2_val / len(val_loader)
    dice3 = dice3_val / len(val_loader)
    return {'loss': loss, 'dice1': dice1, 'dice2': dice2, 'dice3': dice3}


def train(model,optimizer,scheduler,criterion,train_loader,
          val_loader,epochs,device,train_log,valid_loss_min=999.0): #train_log保存训练日志的文件路径，初始化最小验证损失值为999.0
    for e in range(epochs): #遍历训练轮数
        # train for epoch  训练模型，获取损失和dice系数
        train_metrics = train_loop(model,optimizer,scheduler,criterion,train_loader,device,e)
        # eval for epoch
        val_metrics = val_loop(model,criterion,val_loader,device)
        info1 = "Epoch:[{}/{}] train_loss: {:.3f} valid_loss: {:.3f} ".format(e+1,epochs,train_metrics["loss"],val_metrics["loss"])  #生成训练日志第一行信息
        info2 = "Train--ET: {:.3f} TC: {:.3f} WT: {:.3f} ".format(train_metrics['dice1'],train_metrics['dice2'],train_metrics['dice3'])
        info3 = "Valid--ET: {:.3f} TC: {:.3f} WT: {:.3f} ".format(val_metrics['dice1'],val_metrics['dice2'],val_metrics['dice3'])
        print(info1)
        print(info2)
        print(info3)
        with open(train_log, 'a') as f:  #打开训练日志文件
            f.write(info1 + '\n' + info2 + ' ' + info3 + '\n')  #训练日志写入文件

        if not os.path.exists(args.save_path): #保存模型的路径不存在，则创建该路径
            os.makedirs(args.save_path)
        save_file = {"model": model.state_dict(),  #保存模型和优化器的状态字典
                     "optimizer": optimizer.state_dict()}
        if val_metrics['loss'] < valid_loss_min:  #若当前验证损失小于最小验证损失，更新最小验证损失值
            valid_loss_min = val_metrics['loss']
            torch.save(save_file, 'results/unet.pth')
        else:
            torch.save(save_file,os.path.join(args.save_path, 'checkpoint{}.pth'.format(e+1)))
    print("Finished Training")


def main(args):
    torch.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed_all(args.seed)  # 为所有的GPU设置种子，以使得结果是确定的

    torch.backends.cudnn.deterministic = True    #启用确定性算法，保证每次结果相同
    torch.backends.cudnn.benchmark = True       #启用cuDNN的自动调优功能
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'    #设置使用的GPU设备编号
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data info 定义训练、验证、测试三个数据集对象
    patch_size = (160,160,128)  #定义数据集中每个样本的大小
    # 定义训练对象
    train_dataset = BraTS(args.data_path,args.train_txt,transform=transforms.Compose([
        RandomRotFlip(),   #数据集路径      训练集文件路径
        RandomCrop(patch_size),
        GaussianNoise(p=0.1),
        ToTensor()]))
    # 定义验证对象
    val_dataset = BraTS(args.data_path,args.valid_txt,transform=transforms.Compose([
        CenterCrop(patch_size),         #验证集文件路径
        ToTensor()]))
    # 定义测试对象
    test_dataset = BraTS(args.data_path,args.test_txt,transform=transforms.Compose([
        CenterCrop(patch_size),         #测试集文件路径
        ToTensor()]))

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=0,   # num_worker=4
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False,
                            pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False,
                             pin_memory=True)

    print("using {} device.".format(device))
    print("using {} images for training, {} images for validation.".format(len(train_dataset), len(val_dataset)))
    # img,label = train_dataset[0]

    # 1-坏疽(NT,necrotic tumor core),2-浮肿区域(ED,peritumoral edema),4-增强肿瘤区域(ET,enhancing tumor)
    # 评价指标：ET(label4),TC(label1+label4),WT(label1+label2+label4)
    model = UNet(in_channels=4, num_classes=4).to(device)
    criterion = Loss(n_classes=4, weight=torch.tensor([0.2, 0.3, 0.25, 0.25])).to(device)
    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=0, weight_decay=5e-4) #模型、动量参数、初始学习率、权重衰减参数
    scheduler = cosine_scheduler(base_value=args.lr, final_value=args.min_lr, epochs=args.epochs,
                                 niter_per_ep=len(train_loader), warmup_epochs=args.warmup_epochs, start_warmup_value=5e-4)
    #定义学习率调度器对象，初始、最小、总训练轮数、每轮训练迭代数、预热轮数、预热初始学习率

    # 加载训练模型
    if os.path.exists(args.weights):
        weight_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(weight_dict['model'])
        optimizer.load_state_dict(weight_dict['optimizer'])
        print('Successfully loading checkpoint.')

    train(model, optimizer, scheduler, criterion, train_loader,
          val_loader, args.epochs, device, train_log=args.train_log)

    metrics1 = val_loop(model, criterion, train_loader, device)
    metrics2 = val_loop(model, criterion, val_loader, device)
    metrics3 = val_loop(model, criterion, test_loader, device)

    # 最后再评价一遍所有数据，注意，这里使用的是训练结束的模型参数
    print("Train -- loss: {:.3f} ET: {:.3f} TC: {:.3f} WT: {:.3f}".format(metrics1['loss'], metrics1['dice1'],metrics1['dice2'], metrics1['dice3']))
    print("Valid -- loss: {:.3f} ET: {:.3f} TC: {:.3f} WT: {:.3f}".format(metrics2['loss'], metrics2['dice1'], metrics2['dice2'], metrics2['dice3']))
    print("Test  -- loss: {:.3f} ET: {:.3f} TC: {:.3f} WT: {:.3f}".format(metrics3['loss'], metrics3['dice1'], metrics3['dice2'], metrics3['dice3']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.004)
    parser.add_argument('--min_lr', type=float, default=0.002)
    parser.add_argument('--data_path', type=str, default='../BraTS2021/archive/dataset')
    parser.add_argument('--train_txt', type=str, default="../BraTS2021/archive/train.txt")
    parser.add_argument('--valid_txt', type=str, default='../BraTS2021/archive/valid.txt')
    parser.add_argument('--test_txt', type=str, default='../BraTS2021/archive/test.txt')
    parser.add_argument('--train_log', type=str, default='results/unet.txt')
    parser.add_argument('--weights', type=str, default='results/unet.pth')
    parser.add_argument('--save_path', type=str, default='checkpoint/unet')

    args = parser.parse_args()

    main(args)
