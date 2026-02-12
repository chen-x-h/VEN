from __future__ import division, print_function

import torch
import torch.nn as nn


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


class Empty_Decoder(nn.Module):
    def __init__(self, ds=True):
        super(Empty_Decoder, self).__init__()
        self.deep_supervision = ds


if __name__ == '__main__':
    ran = torch.rand([1, 1, 160, 160]) * 98
    ran = ran.cuda()
    r = one_hot_tran(ran, 98)
    # How to use 3d mask rle
    '''
    img_itk = sitk.ReadImage(case_dir)
    mask = sitk.GetArrayFromImage(img_itk)
    mask_ = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
    pros = three2twoRle(mask_, dim=2, times=5, nor=True)
    
    mask_re_ = two2threeRle(pros, dim=dims[0], nor=nor)[0:1]
    
    mask_re = mask_re_.squeeze(0).squeeze(0)
    mask_re = mask_re.numpy()
    dice_ = metric.binary.dc(mask_re.astype(np.int8), mask.astype(np.int8))
    print(f'dice: {dice_}')
    '''
    print('end')

