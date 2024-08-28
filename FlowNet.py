import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import random
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader
import os
from torch import optim
import h5py
from Trans_Morph import TransMorph
from torchvision import transforms
from PIL import Image

class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]
        # print(new_locs.shape)
        warp = F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
        return warp

class FlowNet(nn.Module):
    def __init__(self):
        super(FlowNet, self).__init__()
        self.trans_morph = TransMorph(channels=2)
        self.spatial_trans = SpatialTransformer((512, 512))

    def forward(self, flow, moving):
        warp = self.spatial_trans(moving, flow)
        # concatenated_tensor = torch.cat((flow, warp), dim=1)
        _, flow = self.trans_morph(flow)
        back = self.spatial_trans(warp, flow)
        inverse_warp = self.spatial_trans(moving, flow)
        return flow, back, inverse_warp

class RandomMaskAugmentation:
    def __call__(self, sample):
        image = np.array(sample)  # 将PIL Image对象转换为numpy数组

        # 决定生成多少个mask
        num_masks = self.generate_num_masks()

        # 在图像中生成每个mask
        for _ in range(num_masks):
            top = random.randint(0, 504)  # 512 - 8 = 504
            left = random.randint(0, 504)
            image[top:top+8, left:left+8] = 0  # 修改numpy数组的像素值

        return Image.fromarray(image)  # 将修改后的numpy数组转换为PIL Image对象

    def generate_num_masks(self):
        # 根据提供的概率生成随机数量的mask
        prob = random.random()
        if prob < 0.4:
            return 0
        elif prob < 0.7:
            return 1
        elif prob < 0.9:
            return 2
        else:
            return 3
class FlowDataset(Dataset):
    def __init__(self, paths, flow_paths):
        self.paths = paths
        self.flow_paths = flow_paths
        self.transform_append = transforms.Compose([
            transforms.RandomAffine(degrees=(-3, 3), translate=(0.03, 0.03), scale=(0.97, 1.03),
                                    shear=(-3, 3)),
            RandomMaskAugmentation()
        ])

    def __getitem__(self, index):
        nii_path = self.paths[index]

        image = sitk.GetArrayFromImage(sitk.ReadImage(nii_path))
        numbers = [0, 1, 2, 3, 4, 5, 6]
        # numbers = [2, 4]
        chosen_number = random.choice(numbers)
        moving_image = Image.fromarray(image[chosen_number])
        moving_image = self.transform_append(moving_image)
        moving_image = np.array(moving_image)
        moving_image = torch.from_numpy(moving_image).unsqueeze(0).to('cuda')
        # moving_image = torch.from_numpy(image[chosen_number]).unsqueeze(0).to('cuda')
        random_flow_name = random.choice(self.flow_paths)
        with h5py.File(random_flow_name, 'r') as f:
            dataset = f['flow']
            flow_data = dataset[:]
        flow_data = torch.from_numpy(flow_data).to('cuda')
        # print(flow_data.shape)
        return flow_data, moving_image.float()

    def __len__(self):
        return len(self.paths)


def main():
    batch_size = 32
    train_folder = 'train'
    model_name = 'flow_net.pt'
    # ofg_epoch = 10
    lr = 0.005
    num_epochs = 1024
    model = FlowNet().to('cuda')
    # print(model)

    if os.path.exists(model_name):
        model = torch.load(model_name)

    nii_files = [os.path.join(root, file) for root, dirs, files in os.walk(train_folder) for file in files if
                 file.endswith(".nii")]
    h5_files = [os.path.join(root, file) for root, dirs, files in os.walk('flow') for file in files if
                file.endswith(".h5")]
    # 设置随机种子
    random_seed = 1024
    random.seed(random_seed)
    random.shuffle(nii_files)
    #0.7做训练集，0.3做验证集
    train_num = int(0.7*len(nii_files))
    train_files = nii_files[:train_num]

    validate_files = nii_files[train_num:]

    train_dataset = FlowDataset(train_files, h5_files)
    val_dataset = FlowDataset(validate_files, h5_files)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3,
                                  persistent_workers=True, prefetch_factor=3)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    criterion_mae = nn.L1Loss()

    best_val_loss = float('inf')
    ofg_epoch = 11
    img_size = (512, 512)
    # 训练模型
    for epoch in range(num_epochs):
        # adjust_learning_rate(optimizer, epoch, 23, 0.1)
        # 训练阶段
        mean_train_loss = 0.0
        step_num = 0
        for flow, moving in train_dataloader:
            optimizer.zero_grad()
            back_flow, back_moving, inverse_moving = model(flow, moving)
            # back_moving = torch.where(back_moving == 0, moving, back_moving)
            loss = criterion_mae(moving, back_moving)
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
        with torch.no_grad():
            for flow, moving in val_dataloader:
                back_flow, back_moving, inverse_moving = model(flow, moving)
                back_moving = torch.where(back_moving == 0, moving, back_moving)
                val_loss = criterion_mae(moving, back_moving)
                mean_val_loss += val_loss.item()

        mean_val_loss = mean_val_loss / len(val_dataloader)
        print("Epoch: %d, validate loss: %1.5f" % (epoch, mean_val_loss))
        # 如果当前模型比之前的模型性能更好，则保存当前模型
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            print('best_val_loss:' + str(best_val_loss) + ' saving model:' + model_name)
            torch.save(model, model_name)

if __name__ == "__main__":
    main()