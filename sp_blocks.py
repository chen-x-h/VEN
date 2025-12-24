from __future__ import division, print_function

import torch
import torch.nn as nn


class DropBlock3D(nn.Module):
    """
    3D DropBlock: 随机丢弃连续的 3D 块（立方体区域）
    参考论文: DropBlock: A regularization method for convolutional networks (CVPR 2018)
    扩展到 3D 空间 (D, H, W)
    """

    def __init__(self, drop_prob=0.1, block_size=3):
        """
        Args:
            drop_prob (float): 在训练期间要丢弃的特征的总比例
            block_size (int): 要丢弃的块的最大尺寸（在 depth, height, width 上）
        """
        super(DropBlock3D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        """
        x: (B, C, D, H, W), 例如 (batch, channel, depth, height, width)
        """
        # if not self.training or self.drop_prob == 0.:
        if self.drop_prob == 0.:
            return x

        B, C, D, H, W = x.shape

        # 计算 gamma: 每个位置成为 mask 起点的概率
        # 参考 DropBlock 论文公式
        gamma = (self.drop_prob / (self.block_size ** 3)) * \
                (D * H * W) / ((D - self.block_size + 1) * (H - self.block_size + 1) * (W - self.block_size + 1))

        # 生成随机噪声
        noise = torch.rand(B, C, D, H, W, device=x.device)  # 每个点一个随机数
        mask = noise < gamma  # 初步标记可能的起点

        # 使用 3D 最大池化扩展成 block_size x block_size x block_size 的块
        # kernel_size 和 stride=1，padding=block_size//2 保证输出尺寸不变
        mask = mask.float()
        mask = torch.max_pool3d(
            mask,
            kernel_size=(self.block_size, self.block_size, self.block_size),
            stride=1,
            padding=self.block_size // 2
        )

        # 将 mask 裁剪为 0 或 1
        mask = torch.clamp(mask, 0, 1)
        mask = 1. - mask  # 1 表示保留，0 表示丢弃
        mask *= torch.rand(mask.shape, device=x.device)
        x = x * mask

        # # 推理时也需要归一化，但训练时通过期望值自动调整
        # # 使用期望保留比例进行归一化（可选，也可在训练时用 scale）
        # scale = mask.numel() / mask.sum()  # total_elements / kept_elements
        # x = x * mask * scale  # scale 保证输出期望不变

        return x


class Pro_Block(nn.Module):
    def __init__(self, ch_in, reduction=4, wop=False):
        super(Pro_Block, self).__init__()
        # 计算投影
        # self.avg_pools = nn.ModuleList()
        self.wop = wop
        if wop:
            print('消融：不带投影')
        self.fcs = nn.ModuleList()
        for i in range(3):
            # shape_ = [None] * 3
            # shape_[i] = 1
            # self.avg_pools.append(
            #     nn.AdaptiveAvgPool3d(shape_)
            # )
            self.fcs.append(
                nn.Sequential(
                    nn.Conv3d(ch_in, ch_in // reduction, kernel_size=1, bias=False),
                    # nn.ReLU(inplace=True),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv3d(ch_in // reduction, ch_in, kernel_size=1, bias=False),
                    # nn.Sigmoid()
                )
            )
        self.sig = nn.Sequential(
            nn.Conv3d(ch_in * 3, ch_in, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.had_record = True

    def forward(self, x):
        dim_pro = []
        for i in range(3):
            # TODO， mean？
            if not self.wop:
                pro = torch.mean(x, dim=(i + 2)).unsqueeze(i + 2)
                pro = pro.expand_as(x)
                pro = pro + x
            else:
                pro = x
            # pro = self.avg_pools[i](x)
            pro = self.fcs[i](pro)
            if not self.had_record:
                import numpy as np
                import SimpleITK as sitk
                path_ = '/home/chenxianhao/Liver/Liver_Vessel_Seg/isimg/'
                image = pro.squeeze(0).squeeze(0).cpu().numpy()
                img_itk = sitk.GetImageFromArray(image.astype(np.float32))
                sitk.WriteImage(img_itk, path_ + f'pab0_{i}.nii.gz')
                print('---' + f'pab0_{i}.nii.gz')
            dim_pro.append(pro)

        y = torch.concatenate(dim_pro, dim=1)
        y = self.sig(y)

        if not self.had_record:
            import numpy as np
            import SimpleITK as sitk
            path_ = '/home/chenxianhao/Liver/Liver_Vessel_Seg/isimg/'
            image = y.squeeze(0).squeeze(0).cpu().numpy()
            img_itk = sitk.GetImageFromArray(image.astype(np.float32))
            sitk.WriteImage(img_itk, path_ + f'pab1.nii.gz')
            print('---' + f'pab1.nii.gz')
        self.had_record = True

        # return x * y + x
        return y


class Neigh_Block(nn.Module):
    """
    可以使分割出来的各个部分相互影响，扩大进行分割的感受野
    # TODO 轻量化
    """

    def __init__(self, in_chn, class_num=2, kernel=3, strike=1, padding=1, dil_times=4, conv_3d=nn.Conv3d):
        super(Neigh_Block, self).__init__()
        self.dil_times = dil_times
        self.class_num = class_num
        self.in_chn = in_chn
        k = 3

        self.mean_pool = Mean_Pool(k=k)
        self.max_pool = Dil_Pool(k=k)

        self.finals = nn.ModuleList()
        for i in range(3):
            self.finals.append(
                nn.Sequential(
                    # TODO
                    # act(),
                    conv_3d(in_chn, in_chn, kernel_size=3, padding=1),
                    nn.InstanceNorm3d(in_chn, eps=1e-5, affine=True),
                    nn.LeakyReLU(inplace=True),
                    # nn.Sigmoid()
                )
            )

        self.final_conv = nn.Sequential(
            nn.Conv3d(in_chn, in_chn, kernel_size=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x, change_param=None):

        if change_param is None:
            change_param = self.dil_times
        else:
            change_param = int(change_param * self.dil_times)
            # print(change_param, "debug")
        mid = x
        # TODO
        mid_mean = x.detach().clone()
        mid_max = x.detach().clone()

        out_mean = x.clone()
        out_max = x.clone()

        for i in range(change_param):
            mid_mean = self.mean_pool(mid_mean)
            mid_max = self.max_pool(mid_max)
            out_mean += mid_mean
            out_max += mid_max
        out_mean /= change_param + 1
        out_max /= change_param + 1
        out = self.finals[0](mid) + self.finals[1](out_max) + self.finals[2](out_mean)

        # if self.pro:
        #     out = self.pro_block(out)
        out = self.final_conv(out)

        return out


class Neigh_BlockV2(nn.Module):
    """
    可以使分割出来的各个部分相互影响，扩大进行分割的感受野
    # TODO 轻量化
    """

    def __init__(self, in_chn, dil_times=3, mid_chn=2):
        super(Neigh_BlockV2, self).__init__()
        self.dil_times = dil_times
        self.in_chn = in_chn
        k = 3

        self.mean_pool = Mean_Pool(k=k)
        self.max_pool = Dil_Pool(k=k)
        self.down_conv = nn.Sequential(
            nn.Conv3d(in_chn, mid_chn, kernel_size=1)
        )

        self.finals = nn.ModuleList()
        for i in range(3):
            self.finals.append(
                nn.Sequential(
                    # TODO
                    # act(),
                    nn.Conv3d(mid_chn, in_chn, kernel_size=1),
                    nn.InstanceNorm3d(in_chn, eps=1e-5, affine=True),
                    nn.LeakyReLU(inplace=True),
                    # nn.Sigmoid()
                )
            )

        self.final_conv = nn.Sequential(
            # conv_3d(in_chn * 3, in_chn * 3, kernel_size=3, padding=1),
            nn.Conv3d(in_chn, in_chn, kernel_size=1),
            nn.InstanceNorm3d(in_chn, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
            # nn.LeakyReLU(inplace=True),
        )

    def forward(self, x, change_param=None):

        if change_param is None:
            change_param = self.dil_times
        else:
            change_param = int(change_param * 5.)
            # print(change_param, "debug")
        mid = self.down_conv(x)
        # TODO
        mid_mean = mid.detach()
        mid_max = mid.detach()

        out_mean = mid.clone()
        out_max = mid.clone()

        for i in range(change_param):
            mid_mean = self.mean_pool(mid_mean)
            mid_max = self.max_pool(mid_max)
            out_mean += mid_mean
            out_max += mid_max
        out_mean /= change_param + 1
        out_max /= change_param + 1

        out = self.finals[0](mid) + self.finals[1](out_max) + self.finals[2](out_mean)
        out = self.final_conv(out)

        return out


class Neigh_BlockV3(nn.Module):
    """
    可以使分割出来的各个部分相互影响，扩大进行分割的感受野
    # TODO 轻量化
    """

    def __init__(self, in_chn, dil_times=3, mid_chn=1):
        super(Neigh_BlockV3, self).__init__()
        self.dil_times = dil_times
        self.in_chn = in_chn
        k = 3

        self.mean_pool = Mean_Pool(k=k)
        self.max_pool = Dil_Pool(k=k)

        # self.down_conv = nn.Sequential(
        #     nn.Conv3d(in_chn, mid_chn, kernel_size=1),
        #     nn.InstanceNorm3d(mid_chn, eps=1e-5, affine=True),
        #     nn.LeakyReLU(inplace=True),
        # )

        mid_chn = in_chn
        self.finals = nn.ModuleList()
        for i in range(3):
            self.finals.append(
                nn.Sequential(
                    # TODO
                    # act(),
                    nn.Conv3d(mid_chn, in_chn, kernel_size=1),
                    nn.InstanceNorm3d(in_chn, eps=1e-5, affine=True),
                    nn.LeakyReLU(inplace=True),
                    # nn.Sigmoid()
                )
            )

        self.final_conv = nn.Sequential(
            # conv_3d(in_chn * 3, in_chn * 3, kernel_size=3, padding=1),
            # nn.Conv3d(in_chn, in_chn, kernel_size=1),
            nn.Conv3d(in_chn, in_chn, kernel_size=3, padding=1, groups=in_chn),
            nn.InstanceNorm3d(in_chn, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_chn, in_chn, kernel_size=1),
            nn.InstanceNorm3d(in_chn, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
            # nn.LeakyReLU(inplace=True),
        )

        self.had_record = True

    def forward(self, x, change_param=None):

        if change_param is None:
            change_param = self.dil_times
        else:
            change_param = int(change_param * 5.)
            # print(change_param, "debug")
        # mid = self.down_conv(x)
        mid = x.clone()
        # TODO
        mid_mean = mid.detach()
        mid_max = mid.detach()

        out_mean = mid.clone()
        out_max = mid.clone()

        if not self.had_record:
            import numpy as np
            import SimpleITK as sitk
            path_ = '/home/chenxianhao/Liver/Liver_Vessel_Seg/isimg/'
            image = x.squeeze(0).squeeze(0).cpu().numpy()
            img_itk = sitk.GetImageFromArray(image.astype(np.float32))
            sitk.WriteImage(img_itk, path_ + f'nvpb_0.nii.gz')
            print('---' + f'nvpb_0.nii.gz')

        for i in range(change_param):
            mid_mean = self.mean_pool(mid_mean)
            mid_max = self.max_pool(mid_max)
            out_mean += mid_mean
            out_max += mid_max

            if not self.had_record:
                import numpy as np
                import SimpleITK as sitk
                path_ = '/home/chenxianhao/Liver/Liver_Vessel_Seg/isimg/'
                image = (out_max / (i + 2)).squeeze(0).squeeze(0).cpu().numpy()
                img_itk = sitk.GetImageFromArray(image.astype(np.float32))
                sitk.WriteImage(img_itk, path_ + f'nvpb_max_{i + 1}.nii.gz')
                image = (out_mean / (i + 2)).squeeze(0).squeeze(0).cpu().numpy()
                img_itk = sitk.GetImageFromArray(image.astype(np.float32))
                sitk.WriteImage(img_itk, path_ + f'nvpb_mean_{i + 1}.nii.gz')
                print('---' + f'nvpb_{i + 1}.nii.gz')
        out_mean /= change_param + 1
        out_max /= change_param + 1

        out = self.finals[0](mid) + self.finals[1](out_max) + self.finals[2](out_mean)
        out = self.final_conv(out)

        if not self.had_record:
            import numpy as np
            import SimpleITK as sitk
            path_ = '/home/chenxianhao/Liver/Liver_Vessel_Seg/isimg/'
            image = out.squeeze(0).squeeze(0).cpu().numpy()
            img_itk = sitk.GetImageFromArray(image.astype(np.float32))
            sitk.WriteImage(img_itk, path_ + f'nvpb_f.nii.gz')
            print('---' + f'nvpb_f.nii.gz')

        if not self.had_record:
            self.had_record = True

        return out


class Mean_Pool(nn.Module):
    def __init__(self, k=3, padding_in=False):
        super(Mean_Pool, self).__init__()
        self.k = k
        self.padding_in = padding_in
        self.avg_pool = nn.AvgPool3d(kernel_size=k, stride=1, padding=0)
        if padding_in:
            self.avg_pool = nn.AvgPool3d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        if self.padding_in:
            return self.avg_pool(x)
        x_dil = nn.functional.pad(x, pad=[self.k // 2] * 6, mode='reflect')
        x_dil = self.avg_pool(x_dil)
        return x_dil


class Dil_Pool(nn.Module):
    def __init__(self, k=3, padding_in=False):
        super(Dil_Pool, self).__init__()
        self.k = k
        self.max_pool = nn.MaxPool3d(kernel_size=k, stride=1, padding=0)
        self.padding_in = padding_in
        if padding_in:
            self.max_pool = nn.MaxPool3d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        if self.padding_in:
            return self.max_pool(x)
        x_dil = nn.functional.pad(x, pad=[self.k // 2] * 6, mode='reflect')
        x_dil = self.max_pool(x_dil)
        return x_dil


class Dil_Pool2d(nn.Module):
    def __init__(self, k=3, padding_in=False):
        super(Dil_Pool2d, self).__init__()
        self.k = k
        self.max_pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=0)
        self.padding_in = padding_in
        if padding_in:
            self.max_pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        if self.padding_in:
            return self.max_pool(x)
        x_dil = nn.functional.pad(x, pad=[self.k // 2] * 4, mode='reflect')
        x_dil = self.max_pool(x_dil)
        return x_dil


class L_Conv3d(nn.Module):
    # lite conv3d
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(L_Conv3d, self).__init__()
        # 参考shuffle net，对每个通道做些处理，学习
        # 引入个轻量级mamba进来？对一个通道应该不会很慢？
        # 逐通道卷积转为逐通道处理

        self.depth_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, padding=padding,
                      groups=in_channels, stride=stride),
            nn.InstanceNorm3d(in_channels, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True)
        )
        # self.depth_conv = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, padding=padding,
        #                             groups=in_channels, stride=stride)
        # 在通道卷积之前进行mamba处理或者之后
        # 替换掉逐点卷积
        self.point_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)

        x = self.point_conv(x)
        return x


class L_Conv2d(nn.Module):
    # lite conv3d
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(L_Conv2d, self).__init__()
        # 参考shuffle net，对每个通道做些处理，学习
        # 引入个轻量级mamba进来？对一个通道应该不会很慢？
        # 逐通道卷积转为逐通道处理

        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding,
                                    groups=in_channels, stride=stride)
        # 在通道卷积之前进行mamba处理或者之后
        # 替换掉逐点卷积
        self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)

        x = self.point_conv(x)
        return x


def one_hot_tran(t, num_class=2):
    """
    B, 1, _ -> B, C, _
    """
    shape = list(t.shape)
    shape[1] = num_class
    x_onehot = torch.zeros(shape, device=t.device)
    t_long = t.long()
    t_long = torch.clamp(t_long, min=0, max=num_class - 1)
    x_onehot.scatter_(1, t_long, 1)
    return x_onehot.float()


def logit_tran(t):
    """
    B, C, _ -> B, 1, _
    """
    return torch.argmax(torch.softmax(t, dim=1), dim=1).unsqueeze(1)


def color_text(t):
    return f'\033[1;33m {t} \033[0m'


def save_vis_image(image, name, label=False):
    """
    # save_vis_image(x[0], 'x2d.nii.gz')
    # save_vis_image(d3_data['gt'][0], 'gt.nii.gz', label=True)
    # save_vis_image(d3_data['full_data'][0], 'f_img.nii.gz')
    # save_vis_image(d3_data['full_seg'][0], 'f_seg.nii.gz', label=True)
    """
    import numpy as np
    import SimpleITK as sitk
    if name.find('.') == -1:
        name += '.nii.gz'
    path_ = '/home/chenxianhao/Liver/Liver_Vessel_Seg/isimg/'
    print(image.shape)
    image = image.cpu().numpy()
    type_ = np.float32
    if label:
        type_ = np.int16
    img_itk = sitk.GetImageFromArray(image.astype(type_))
    sitk.WriteImage(img_itk, path_ + name)


def random_change_mask(masks,
                       hole_prob=0.2,
                       max_hole_size=8,
                       morph_prob=0.5,
                       max_kernel=3,
                       blur_prob=0.3,
                       sigma_range=(0.5, 1.5),
                       no_change_rate=0.5
                       ):
    """
    对批量的二值掩码进行随机扰动 [B, 1, H, W]。

    Args:
        masks (torch.Tensor)
        hole_prob: 挖洞概率
        max_hole_size: 洞的最大半径
        morph_prob: 形态学操作概率
        max_kernel: 形态学结构元素最大尺寸 (会生成奇数 kernel)
        blur_prob: 模糊+重二值化概率
        sigma_range: 高斯模糊 sigma 范围
        no_change_rate: 完全不变的概率
    Returns:
        torch.Tensor: 扰动后的掩码, shape [B, 1, H, W], dtype byte
    """

    return masks


def random_loss_mask(mask, min_th=0., max_th=0.3):
    """
    随机丢弃mask
    """
    return mask


def random_open_mask(mask, min_th=0., max_th=0.3, times=1):
    """
    随机开运算mask
    """
    return mask


# 参考行程编码，假定一个最多几个行程？
def three2twoRle_single(mask, dim=2, times=5, nor=True):
    """
    times 足够大的时候无损，一般来说到7就无损了，但是2也够用了
    """
    shape_ = mask.shape
    device_ = mask.device
    # TODO 要在顶部和底部加个0，不然通天柱始终去不掉
    tmp_shape = list(shape_)
    tmp_shape[dim] = 1
    tmp_s = torch.zeros(tmp_shape, device=mask.device)
    mask = torch.concatenate([tmp_s, mask, tmp_s], dim=dim)
    tmp_shape[dim] = shape_[dim] + 2
    rets = []
    x_indices_shape = [1] * 5
    x_indices_shape[dim] = -1
    x_indices = torch.arange(tmp_shape[dim]).view(x_indices_shape).to(device=device_)
    # last_top = None
    last_low = None
    for i in range(times):
        t_low = torch.argmax(mask, dim=dim)
        mask[x_indices < t_low] = 1
        t_top = torch.argmax(-mask, dim=dim)
        t_top = torch.where(t_top > t_low, t_top, t_low)

        mask[x_indices <= t_top] = 0

        # TODO 差值模式
        # if i > 0:
        #     t_low_ = (t_low - last_low)
        # else:
        #     t_low_ = t_low
        # t_top_ = (t_top - t_low)

        # last_top = t_top
        # last_low = t_low
        t_top_ = t_top
        t_low_ = t_low

        if nor:
            t_low_ = t_low_.to(dtype=torch.float) / (tmp_shape[dim])
            t_top_ = t_top_.to(dtype=torch.float) / (tmp_shape[dim])
        else:
            t_low_ = t_low_.to(dtype=torch.int32)
            t_top_ = t_top_.to(dtype=torch.int32)

        rets.append(t_low_)
        rets.append(t_top_)

        # save_img(mask, f'./mask_{i}.nii.gz')
        #
        # print(mask.sum())
    out = [
        torch.concatenate(rets, dim=1),
        shape_
    ]
    # rets.append(shape_)
    return out


def three2twoRle(mask, dim=2, times=5, nor=True):
    """
    输入尺寸为 b, 1, x, y, z
    batch里的size一致
    返回压缩后的张量和原先的尺寸
    """
    b_mask = []
    b_size = None
    for i in range(mask.shape[0]):
        mask_, size_ = three2twoRle_single(mask[i:i+1], dim=dim, times=times, nor=nor)
        b_mask.append(mask_)
        b_size = size_
    out = [
        torch.concatenate(b_mask, dim=0),
        b_size
    ]
    return out


def two2threeRle_single(pros, dim=2, nor=True):
    shape_ = list(pros[-1])
    shape_[0] = 1
    device_ = pros[0].device
    tmp_shape = list(shape_)
    tmp_shape[dim] += 2
    mask = torch.zeros(size=tmp_shape, device=device_)
    x_indices_shape = [1] * 5
    x_indices_shape[dim] = -1
    x_indices = torch.arange(tmp_shape[dim]).view(x_indices_shape).to(device=device_)
    # in_2d = pros[0].float()
    in_2d = pros[0]
    # last_top = None
    last_low = None
    for i in range(0, (in_2d.shape[1]) // 2):
        t_low = in_2d[:, i * 2:i * 2 + 1].unsqueeze(dim)
        t_top = in_2d[:, i * 2 + 1:i * 2 + 2].unsqueeze(dim)
        # 差值模式
        # if i > 0:
        #     t_low = t_low + last_low
        # t_top = t_top + t_low
        # # last_top = t_top
        # last_low = t_low

        if nor:
            t_low = t_low * (tmp_shape[dim])
            t_top = t_top * (tmp_shape[dim])

        mask[((x_indices >= t_low) & (x_indices < t_top))] = 1
        # mask[((x_indices == t_low) & (x_indices == t_top))] = 0

    mask = torch.narrow(mask, dim, 1, mask.size(dim) - 2)

    mask[mask > 0] = 1
    mask[mask < 1] = 0

    return mask


def two2threeRle(pros, dim=2, nor=True):
    rle_mask = pros[0]
    shapes = pros[1]
    batch_size = rle_mask.shape[0]
    res = []
    for i in range(batch_size):
        mask = two2threeRle_single([rle_mask[i:i+1], shapes], dim=dim, nor=nor)
        res.append(mask)
    return torch.concatenate(res, dim=0)


class Sim_vessel_Block(nn.Module):
    def __init__(self):
        super(Sim_vessel_Block, self).__init__()
        self.dilate_pool = Dil_Pool(3)

    def forward(self, x, open_times=2, dil_times=0):
        x = x.float()
        # 开运算
        image_s = -x.clone()
        for i in range(open_times):
            image_s = self.dilate_pool(image_s)
        image_s = -image_s
        for i in range(open_times):
            image_s = self.dilate_pool(image_s)

        # main_vessel = image_s
        sim_vessel = x - image_s
        # sim_vessel[sim_vessel > 0] = 1
        # sim_vessel[sim_vessel < 1e-8] = 0
        main_vessel = x - sim_vessel
        # main_vessel[main_vessel > 0] = 1
        # main_vessel[main_vessel < 1e-8] = 0
        for i in range(dil_times):
            sim_vessel = self.dilate_pool(sim_vessel)
        # sim_vessel[sim_vessel > 0] = 1
        # sim_vessel[sim_vessel < 1e-8] = 0
        return sim_vessel, main_vessel


class Empty_Decoder(nn.Module):
    def __init__(self, ds=True):
        super(Empty_Decoder, self).__init__()
        self.deep_supervision = ds


if __name__ == '__main__':
    ran = torch.rand([1, 1, 160, 160]) * 98
    ran = ran.cuda()
    r = one_hot_tran(ran, 98)
    print('end')
