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


def main():
    os.makedirs('test', exist_ok=True)
    train_folder = 'train'
    model_name = 'u_morph.pt'
    model = torch.load(model_name).to('cuda')
    print(model)

    png_files = [os.path.join(root, file) for root, dirs, files in os.walk(train_folder) for file in files if
                 file.endswith(".png")]
    # 设置随机种子
    random_seed = 1024
    random.seed(random_seed)
    random.shuffle(png_files)
    #0.7做训练集，0.3测试集
    train_num = int(0.7*len(png_files))
    test_files = png_files[train_num:]

    to_pil = transforms.ToPILImage()
    for test_file in test_files:
        ct_image = Image.open(test_file)
        ct_image = np.array(ct_image)
        ct_image = torch.from_numpy(ct_image).float()
        ct_image = ct_image.permute(2, 0, 1)
        fixed_image = ct_image[0, :, :]
        fixed_image = to_pil(fixed_image)
        fixed_image_name = os.path.basename(test_file)
        fixed_image_name = fixed_image_name.replace('.png', '_fixed.png')
        fixed_image_name = 'test/' + fixed_image_name
        fixed_image.save(fixed_image_name)
        moving_image = ct_image[1, :, :]
        moving_image = to_pil(moving_image)
        moving_image_name = os.path.basename(test_file)
        moving_image_name = moving_image_name.replace('.png', '_moving.png')
        moving_image_name = 'test/' + moving_image_name
        moving_image.save(moving_image_name)
        ct_image = ct_image.unsqueeze(0).to('cuda')
        (source, ref, warp_source, warp_ref, source_back, ref_back, projection_source, b_spline_source,
         warp_optical_source, warp_optical_ref, optical_source_back, optical_ref_back) = model(ct_image)
        warp_image = warp_ref.squeeze(0).squeeze(0)
        warp_image = to_pil(warp_image)
        warp_image_name = os.path.basename(test_file)
        warp_image_name = warp_image_name.replace('.png', '_warp.png')
        warp_image_name = 'test/' + warp_image_name
        warp_image.save(warp_image_name)


if __name__ == '__main__':
    main()
    # train_folder = 'train'
    # get_mean_std(train_folder)


