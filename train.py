import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch import optim
from torch.utils.data import Dataset, DataLoader
from math import pi, cos
from torchvision import models,transforms
from Trans_Morph import TransMorph
from u_morph import UMorph
import random
from PIL import Image



class LNCC(nn.Module):
    def __init__(self, window_size=7):
        super(LNCC, self).__init__()
        self.window_size = window_size
        self.padding = window_size // 2

    def forward(self, I, J):
        I = I/255.0
        J = J/255.0
        # 计算局部均值
        I_mean = F.avg_pool2d(I, self.window_size, stride=1, padding=self.padding)
        J_mean = F.avg_pool2d(J, self.window_size, stride=1, padding=self.padding)

        # 计算局部标准差
        I_var = F.avg_pool2d(I * I, self.window_size, stride=1, padding=self.padding) - I_mean * I_mean
        J_var = F.avg_pool2d(J * J, self.window_size, stride=1, padding=self.padding) - J_mean * J_mean
        I_std = torch.sqrt(I_var + 1e-5)
        J_std = torch.sqrt(J_var + 1e-5)

        # 计算局部NCC
        ncc = (F.avg_pool2d(I * J, self.window_size, stride=1, padding=self.padding) - I_mean * J_mean) / (
                    I_std * J_std + 1e-5)

        # 计算LNCC损失，最大化NCC等同于最小化 -NCC
        loss = -torch.mean(ncc)
        # 归一化到 0 ~ 1
        loss = (loss + 1) / 2  # 将损失从 [-1, 1] 转换到 [0, 1]
        return loss


class LMI(nn.Module):
    def __init__(self, window_size=7, num_bins=32):
        super(LMI, self).__init__()
        self.window_size = window_size
        self.padding = window_size // 2
        self.num_bins = num_bins

    def forward(self, I, J):
        I = I/255.0
        J = J/255.0
        # 将图像值量化到 [0, num_bins-1]
        I = torch.round(I * (self.num_bins - 1))
        J = torch.round(J * (self.num_bins - 1))

        # 创建直方图的窗口
        joint_hist = self.compute_joint_histogram(I, J)

        # 计算联合概率
        joint_prob = joint_hist / torch.sum(joint_hist)

        # 计算边缘概率
        I_prob = torch.sum(joint_prob, dim=1)
        J_prob = torch.sum(joint_prob, dim=0)

        # 计算互信息
        mutual_info = torch.sum(joint_prob * (torch.log(joint_prob + 1e-5)
                                              - torch.log(I_prob.unsqueeze(1) + 1e-5)
                                              - torch.log(J_prob.unsqueeze(0) + 1e-5)))

        # 归一化0~1
        normalized_loss = torch.sigmoid(-mutual_info)
        return normalized_loss

    def compute_joint_histogram(self, I, J):
        # 初始化联合直方图
        joint_hist = torch.zeros(self.num_bins, self.num_bins).to(I.device)

        # 计算局部联合直方图
        for i in range(self.num_bins):
            for j in range(self.num_bins):
                joint_hist[i, j] = torch.sum((I == i).float() * (J == j).float())

        return joint_hist


class CTDataset(Dataset):
    def __init__(self, ct_image_list, device):
        self.ct_image_list = ct_image_list
        self.transform = transforms.RandomAffine(degrees=3, translate=(0.0125, 0.0125), scale=(0.9875, 1.0125), shear=2)
        self.device = device

    def __len__(self):
        return len(self.ct_image_list)

    def __getitem__(self, idx):
        ct_image_name = self.ct_image_list[idx]
        ct_image = Image.open(ct_image_name)
        ct_image = np.array(ct_image)
        ct_image = torch.from_numpy(ct_image).float()
        # channel_1 = ct_image[:, :, 0:1]
        # # print(channel_1.size())
        # channel_2 = ct_image[:, :, 1:2]
        #
        # # Apply random affine transform to each channel independently
        # transformed_channel_1 = self.transform(channel_1)
        # transformed_channel_2 = self.transform(channel_2)
        #
        #
        # # Stack them back to form a 2-channel image
        # ct_image = torch.stack([transformed_channel_1, transformed_channel_2])
        # ct_image = ct_image[:, :, :, 0]
        # 缩放到 [0, 1] 范围
        # ct_image = ct_image / 255.0
        ct_image = ct_image.permute(2, 0, 1)
        #随机选择两个通道，做配对
        lists = [0, 1]
        random_num = random.choice(lists)
        if random_num:
            ct_image = torch.cat([ct_image[1:2, :, :], ct_image[0:1, :, :]], dim=0)
        return ct_image.to(self.device)


def main():
    batch_size = 8
    train_folder = 'train'
    model_name = 'u_morph.pt'
    # ofg_epoch = 10
    lr = 0.01
    num_epochs = 1024
    model = UMorph().to('cuda')
    # print(model)

    if os.path.exists(model_name):
        model = torch.load(model_name)

    png_files = [os.path.join(root, file) for root, dirs, files in os.walk(train_folder) for file in files if
                 file.endswith(".png")]
    # 设置随机种子
    random_seed = 1024
    random.seed(random_seed)
    random.shuffle(png_files)
    #0.7做训练集，0.3测试集
    train_num = int(0.7*len(png_files))
    train_files = png_files[:train_num]

    test_files = png_files[train_num:]
    #训练集中的0.7做训练集，0.3做验证集
    random.shuffle(train_files)
    validate_num = int(train_num*0.3)
    train_files = train_files[validate_num:]

    validate_files = train_files[:validate_num]

    train_dataset = CTDataset(train_files, 'cuda')
    val_dataset = CTDataset(validate_files, 'cuda')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3,
                                  persistent_workers=True, prefetch_factor=3)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    criterion_lncc = LNCC()
    criterion_lmi = LMI()
    criterion_mae = nn.L1Loss()
    best_val_loss = float('inf')
    img_size = (512, 512)
    # #生成文件夹用来存储flow
    # os.makedirs('flow', exist_ok=True)
    # #加载求取backward flow的模型
    # back_model = torch.load('flow_net.pt').to('cuda')
    # 训练模型
    for epoch in range(num_epochs):
        # adjust_learning_rate(optimizer, epoch, 23, 0.1)
        # 训练阶段
        mean_train_loss = 0.0
        step_num = 0
        for input_tensor in train_dataloader:
            optimizer.zero_grad()
            (source, ref, warp_source, warp_ref, source_back, ref_back, projection_source, b_spline_source,
             warp_optical_source, warp_optical_ref, optical_source_back, optical_ref_back) = model(input_tensor)
            loss_lncc = (criterion_lncc(source, warp_ref) + criterion_lncc(source, source_back) +
                         criterion_lncc(ref, warp_source) + criterion_lncc(ref, ref_back) +
                         criterion_lncc(projection_source, ref) + criterion_lncc(b_spline_source, ref))
            loss_lmi = (criterion_lmi(source, warp_ref) + criterion_lmi(source, source_back) +
                        criterion_lmi(ref, warp_source) + criterion_lmi(ref, ref_back) +
                        criterion_lmi(projection_source, ref) + criterion_lmi(b_spline_source, ref))
            loss_mae = (criterion_mae(source, warp_optical_ref) + criterion_mae(ref, warp_optical_source) +
                        criterion_mae(source, optical_source_back) + criterion_mae(ref, optical_ref_back))/255.0

            loss = loss_lncc + loss_lmi + loss_mae
            loss.backward()  # 计算梯度
            optimizer.step()
            # running_train_loss += loss.item() * inputs.size(0)
            mean_train_loss = (mean_train_loss*step_num + loss.item())/float(step_num + 1)
            step_num = step_num + 1
            print("Epoch: %d, train loss: %1.5f, mean loss: %1.5f, min val loss: %1.5f" % (epoch, loss.item(),
                                                                                           mean_train_loss,
                                                                                           best_val_loss))

        # 验证阶段
        mean_val_loss = 0
        with ((torch.no_grad())):
            for input_tensor in val_dataloader:
                (source, ref, warp_source, warp_ref, source_back, ref_back, projection_source, b_spline_source,
                 warp_optical_source, warp_optical_ref, optical_source_back, optical_ref_back) = model(input_tensor)
                loss_lncc = (criterion_lncc(source, warp_ref) + criterion_lncc(source, source_back) +
                             criterion_lncc(ref, warp_source) + criterion_lncc(ref, ref_back) +
                             criterion_lncc(projection_source, ref) + criterion_lncc(b_spline_source, ref))
                loss_lmi = (criterion_lmi(source, warp_ref) + criterion_lmi(source, source_back) +
                            criterion_lmi(ref, warp_source) + criterion_lmi(ref, ref_back) +
                            criterion_lmi(projection_source, ref) + criterion_lmi(b_spline_source, ref))
                loss_mae = (criterion_mae(source, warp_optical_ref) + criterion_mae(ref, warp_optical_source) +
                            criterion_mae(source, optical_source_back) + criterion_mae(ref, optical_ref_back)) / 255.0

                val_loss = loss_lncc + loss_lmi + loss_mae
                mean_val_loss += val_loss.item()

        mean_val_loss = mean_val_loss / len(val_dataloader)
        print("Epoch: %d, validate loss: %1.5f" % (epoch, mean_val_loss))
        # 如果当前模型比之前的模型性能更好，则保存当前模型
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            print('best_val_loss:' + str(best_val_loss) + ' saving model:' + model_name)
            torch.save(model, model_name)


if __name__ == '__main__':
    main()
    # train_folder = 'train'
    # get_mean_std(train_folder)


