from torch.distributions.normal import Normal
import torch.nn.functional as nnf
import torch.nn.functional as F

import torch
import torch.nn as nn


class UNet(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=3, out_channels=3, kernel_size=3):
        """Initializes U-Net."""

        super(UNet, self).__init__()

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, kernel_size, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, kernel_size, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, kernel_size, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, kernel_size, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size, stride=1, padding=1),
            nn.LeakyReLU(0.1))

        # Initialize weights
        self._init_weights()


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """
        # Encoder
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block2(pool2)
        pool4 = self._block2(pool3)
        pool5 = self._block2(pool4)

        # Decoder
        upsample5 = self._block3(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block4(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block5(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block5(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)

        # Final activation
        return self._block6(concat1)


class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv2d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv2d.weight.shape))
        conv2d.bias = nn.Parameter(torch.zeros(conv2d.bias.shape))
        super().__init__(conv2d)


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
        warp = nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
        return warp

#我们用savitzky_golay_filter模拟B样条滤波
class BSplineFilter(nn.Module):
    def __init__(self):
        super(BSplineFilter, self).__init__()
        # 生成一个5x5的近似三次Savitzky-Golay滤波核
        kernel = torch.tensor([
            [-3, 12, 17, 12, -3],
            [0, -5, -20, -5, 0],
            [4, 16, 26, 16, 4],
            [0, -5, -20, -5, 0],
            [-3, 12, 17, 12, -3]
        ], dtype=torch.float32)
        kernel = kernel / kernel.sum()  # 归一化
        self.kernel = kernel.view(1, 1, 5, 5)

    def forward(self, flow):
        # 对于每个通道应用卷积
        flow_x = flow[:, 0:1, :, :]
        flow_y = flow[:, 1:2, :, :]
        # 假设 flow_x 和 flow_y 都在 GPU 上
        self.kernel = self.kernel.to(flow.device)
        smoothed_flow_x = F.conv2d(flow_x, self.kernel, padding=2)
        smoothed_flow_y = F.conv2d(flow_y, self.kernel, padding=2)

        # 合并两个平滑后的通道
        smoothed_flow = torch.cat([smoothed_flow_x, smoothed_flow_y], dim=1)
        # print(smoothed_flow.size())

        return smoothed_flow


class UMorph(nn.Module):
    def __init__(self, channels=2):
        super(UMorph, self).__init__()
        self.unet = UNet(in_channels=2, out_channels=2, kernel_size=3)
        self.reg_head = RegistrationHead(
            in_channels=2,
            out_channels=10,
            kernel_size=3,
        )
        self.projection_theta = nn.Linear(524288, 9)
        self.projection_flow = nn.Linear(9, 524288)
        self.spatial_trans = SpatialTransformer((512, 512))
        self.bspline_filter = BSplineFilter()

    def optical_warp(self, img, flow):
        B, C, H, W = img.size()
        flow = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]

        # 生成标准网格，使用 'ij' 索引模式
        grid_y, grid_x = torch.meshgrid(torch.arange(0, H, device=img.device),
                                        torch.arange(0, W, device=img.device),
                                        indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1).float()  # [H, W, 2]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]

        # 添加光流位移
        grid = grid + flow

        # 将网格规范化到 [-1, 1]
        grid[:, :, :, 0] = 2.0 * grid[:, :, :, 0] / (W - 1) - 1.0
        grid[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / (H - 1) - 1.0

        return F.grid_sample(img, grid, align_corners=True)

    def forward(self, x):
        source = x[:, 0:1, :, :]
        ref = x[:, 1:2, :, :]

        x_unet = x.clone()
        x_unet = self.unet(x_unet)
        #得到投影变换
        features = x_unet.view(x_unet.size(0), -1)
        theta = self.projection_theta(features)
        projection = self.projection_flow(theta)
        projection = projection.view(-1, 2, 512, 512)
        projection_source = self.spatial_trans(source, projection)

        #得到defrom transform
        flow = self.reg_head(x_unet)
        # print(flow.size())
        warp_source = self.spatial_trans(source, flow[:, 0:2, :, :])
        warp_ref = self.spatial_trans(ref, flow[:, 2:4, :, :])
        source_back = self.spatial_trans(warp_source, flow[:, 2:4, :, :])
        ref_back = self.spatial_trans(warp_ref, flow[:, 0:2, :, :])

        #得到B样条变换
        bspline_flow = self.bspline_filter(flow[:, 4:6, :, :])
        # print(bspline_flow.size())
        b_spline_source = self.spatial_trans(source, bspline_flow)

        #得到光流变换
        warp_optical_source = self.optical_warp(source, flow[:, 6:8, :, :])
        warp_optical_ref = self.optical_warp(ref, flow[:, 8:10, :, :])
        optical_source_back = self.optical_warp(warp_optical_source, flow[:, 8:10, :, :])
        optical_ref_back = self.optical_warp(warp_optical_ref, flow[:, 6:8, :, :])

        return (source, ref, warp_source, warp_ref, source_back, ref_back, projection_source, b_spline_source,
                warp_optical_source, warp_optical_ref, optical_source_back, optical_ref_back)



if __name__ == "__main__":
    # Create the model and a dummy input
    model = UMorph()
    input_tensor = torch.rand(16, 2, 512, 512)  # Batch size of 1

    # Forward pass through the model with the dummy input
    output = model(input_tensor)
