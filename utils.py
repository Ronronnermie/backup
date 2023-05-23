import torch.nn.functional as F
import torch.nn as nn
import torch
from einops import rearrange
import numpy as np


def Dice(output, target, eps=1e-3):
    inter = torch.sum(output * target, dim=(1, 2, 3)) + eps
    union = torch.sum(output, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3)) + eps * 2
    x = 2 * inter / union
    dice = torch.mean(x)  #计算Dice系数的平均值，即在batch维度上求平均。
    return dice


def cal_dice(output, target):
    output = torch.argmax(output, dim=1)
    dice1 = Dice((output == 3).float(), (target == 3).float()) #计算类别为3的Dice系数，即模型输出和真实标签中都为3的像素点的Dice系数。4
    dice2 = Dice(((output == 1) | (output == 3)).float(), ((target == 1) | (target == 3)).float())  #14
    dice3 = Dice((output != 0).float(), (target != 0).float()) #非背景类别的Dice系数，即模型输出和真实标签中非0的像素点的Dice系数。124

    return dice1, dice2, dice3


# 生成一个cosine学习率调度表
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0.): #学习率初始值，最终值，训练轮数，每轮训练迭代总次数，预热轮数预热学习率默认0
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep #计算预热的迭代次数
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class Loss(nn.Module):
    def __init__(self, n_classes, weight=None, alpha=0.5):
        "dice_loss_plus_cetr_weighted"
        super(Loss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight.cuda()
        # self.weight = weight
        self.alpha = alpha  #总损失中的权重比例

    def forward(self, input, target):
        smooth = 0.01  # 防止分母为0
        input1 = F.softmax(input, dim=1)  #对分类维度进行softmax
        target1 = F.one_hot(target, self.n_classes)
        input1 = rearrange(input1, 'b n h w s -> b n (h w s)')  #(batch_size, n_classes, height*width*depth)
        target1 = rearrange(target1, 'b h w s n -> b n (h w s)')  #

        input1 = input1[:, 1:, :]       #去掉第二维的背景类，只保留分割类别的概率分布。
        target1 = target1[:, 1:, :].float()

        # 以batch为单位计算loss和dice_loss，据说训练更稳定，和上面的公式有出入
        # 注意，这里的dice不是真正的dice，叫做soft_dice更贴切
        inter = torch.sum(input1 * target1)
        union = torch.sum(input1) + torch.sum(target1) + smooth
        dice = 2.0 * inter / union

        loss = F.cross_entropy(input, target, weight=self.weight)

        total_loss = (1 - self.alpha) * loss + (1 - dice) * self.alpha

        return total_loss


if __name__ == '__main__':
    torch.manual_seed(3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losser = Loss(n_classes=4, weight=torch.tensor([0.2, 0.3, 0.25, 0.25])).to(device)
    x = torch.randn((2, 4, 16, 16, 16)).to(device)   #用来生成随机数字的tensor 创建随机张量,2是batch_size，4是分类类别，HWD
    y = torch.randint(0, 4, (2, 16, 16, 16)).to(device) #创建一个大小为(2, 16, 16, 16)的随机张量y，随机整数张量，张量中的元素取值范围为[0, 4)
    print(losser(x, y))
