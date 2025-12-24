import copy
import itertools
import json
import math
import os

import numpy as np
import torch
import SimpleITK as sitk
from npnet import UNet3dwRle, Neigh3dPromptWrapv5
import torch.nn.functional as F
from sp_blocks import *

D3_Z_LEN = 16


def get_model(model_id, input_channels=1, output_channels=2, deep_supervision=True, name_only=False, patch_size=None,
              best=False, data_id=None):
    size_ = [16, 320, 320]
    network = None
    if patch_size is not None:
        size_ = patch_size

    if model_id == 135:
        base_net = UNet3dwRle(1, output_channels, ds=True,
                              shape_=size_, times=2, do_merge=False)
        network = Neigh3dPromptWrapv5(input_channels, output_channels, ds=deep_supervision, gt_fre=1,
                                      shape_=size_, base_net=base_net)
    elif model_id == 136:
        feature_chns = [32, 32, 64, 64, 128, 128]
        base_net = UNet3dwRle(1, output_channels, ds=True,
                              shape_=size_, times=2, do_merge=False, feature_chns=feature_chns)
        network = Neigh3dPromptWrapv5(input_channels, output_channels, ds=deep_supervision, gt_fre=1,
                                      shape_=size_, base_net=base_net)
    else:
        raise RuntimeError("error model pick")

    return network


class Predictor:
    def __init__(self, device='cpu'):
        """
        使用说明
        0、初始化
        predictor = Predictor(device='cpu') #可以设定gpu
        predictor.load_model() #可选模型路径

        1、加载图像（建议预先重采样为[0.8,0.8,1.0]，这里有重采样的逻辑，但会稍显复杂，效果也会差一点）
        predictor.load_img(img: np, sitk.GetSpacing())

        2、全局推理 输入图像、输出掩码 (np)
        pred_mask = predictor.pred_all(mask)
        mask可以全0，如果不为0的话会作为提示

        3、提示推理
        pred_mask = predictor.pred(sl, mask)
        sl是当前切片号（已经修改好的），mask是当前的掩码（里面包含修改好的sl层切片掩码），
        只会处理 [sl-8, sl+7]共16层，其中sl所在的掩码不会变

        :param device:
        """
        self.MODEL = None
        self.device = device
        self.img = None
        self.overlap = 0
        self.d3_z_len = D3_Z_LEN
        self.neigh_len = self.d3_z_len // 2
        self.do_mirror = False
        # 预处理配置
        self.max = 500.
        self.min = 0.
        self.base_spacing = [0.8, 0.8, 1.0]
        self.current_spacing = copy.deepcopy(self.base_spacing)
        self.ori_shape = None
        self.need_re = False
        self.re_img = None
        self.resample_shape = None
        # 滑动窗口配置
        self.win_overlap = (64, 64)
        self.window_size = (320, 320)

        self.max_batch = 2

    def CT_normalize(self, nii_data):
        """
        normalize
        Our values currently range from 0 to around 500.
        Anything above 400 is not interesting to us,
        as these are simply bones with different radiodensity.
        """
        max_, min_ = self.max, self.min
        print(f'0-1 norm, {min_}-{max_}')
        nii_data = nii_data.astype(np.float32)

        nii_data = (nii_data - min_) / (max_ - min_)
        nii_data[nii_data > 1] = 1.
        nii_data[nii_data < 0] = 0.
        return nii_data

    def resample_to_torch(self, img, is_label=False, tar_shape=None):
        """
        对 3D 单通道张量 (D, H, W) 进行重采样到目标 spacing。

        Args:
            img (torch.Tensor): 形状为 (D, H, W)
            is_label (bool): 是否为标签（决定插值方式）
            tar_shape: 设定后就转那个

        Returns:
            torch.Tensor: 重采样后的 (D', H', W')
        """
        if img.ndim != 3:
            raise ValueError(f"Expected 3D input (D, H, W), got shape {img.shape}")

        ori_type = img.dtype

        if not getattr(self, 'need_re', False):
            self.resample_shape = img.shape
            return img

        # 添加 batch 和 channel 维度: (1, 1, D, H, W)
        x = img.unsqueeze(0).unsqueeze(0).float()  # → (B=1, C=1, D, H, W)

        if not is_label and self.resample_shape is None:
            # 计算目标空间尺寸
            self.ori_shape = img.shape

            new_shape = [
                int(np.round(self.current_spacing[2] * self.ori_shape[0] / self.base_spacing[2])),
                int(np.round(self.current_spacing[0] * self.ori_shape[1] / self.base_spacing[0])),
                int(np.round(self.current_spacing[1] * self.ori_shape[2] / self.base_spacing[1])),
            ]
            self.resample_shape = new_shape

        # 插值
        mode = 'nearest' if is_label else 'trilinear'
        align_corners = False if is_label else True

        if tar_shape is None:
            tar_shape = self.resample_shape

        if align_corners:
            x_resampled = F.interpolate(
                x,
                size=tar_shape,
                mode=mode,
                align_corners=align_corners
            )  # (1, 1, D', H', W')
        else:
            x_resampled = F.interpolate(
                x,
                size=tar_shape,
                mode=mode
            )  # (1, 1, D', H', W')

        # 移除 batch 和 channel 维度 → (D', H', W')
        result = x_resampled.squeeze(0).squeeze(0).to(dtype=ori_type)

        # 缓存重采样后的图像（非标签）
        if not is_label:
            self.re_img = result.detach()

        return result

    def load_img(self, img, spacing):
        """
        这里需要预处理
        重采样、裁剪、归一化
        :param img: np: (z,x,y)
        :param spacing: sitk读取的spacing，顺序是[x,y,z]
        :return:
        """
        self.current_spacing = spacing
        print(f'current spacing{spacing}, base sp{self.base_spacing}')

        # 判断做不做重采样
        self.need_re = False
        for i in range(len(spacing)):
            # 判断是否需要重采样
            if math.fabs(spacing[i] - self.base_spacing[i]) >= 1e-1:
                self.need_re = True
                # TODO 必须要事先重采样！！
                print('error need re!')
                break

        if np.max(img) > 1:
            img = self.CT_normalize(img)
        self.img = torch.from_numpy(img).float().to(device=self.device)

        if self.need_re:
            print('do resample')
            self.resample_to_torch(self.img)

    def img_resample(self, itk, new_spacing=None, label_re=False, new_shape=None):
        """
        重采样一个sitk对象
        :param new_shape: 设置来确保还原尺寸一致
        :param itk:
        :param new_spacing:
        :param label_re:
        :return:
        """
        if new_spacing is None:
            new_spacing = self.base_spacing
        original_spacing = itk.GetSpacing()
        # print(original_spacing)
        original_size = itk.GetSize()
        # print(data.GetOrigin(), data.GetDirection())
        if new_shape is None:
            new_shape = [
                int(np.round(original_spacing[0] * original_size[0] / new_spacing[0])),
                int(np.round(original_spacing[1] * original_size[1] / new_spacing[1])),
                int(np.round(original_spacing[2] * original_size[2] / new_spacing[2])),
            ]
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(sitk.sitkLinear)
        if label_re:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        # TODO 应该没有漏？
        resample.SetDefaultPixelValue(0)
        resample.SetOutputSpacing(new_spacing)
        resample.SetOutputDirection(itk.GetDirection())
        resample.SetOutputOrigin(itk.GetOrigin())
        resample.SetSize(new_shape)
        new_itk = resample.Execute(itk)
        return new_itk

    def extract_patches_3d(self, tensor: torch.Tensor) -> tuple[
        torch.Tensor, list[tuple[int, int]], tuple[int, int], tuple[int, int]
    ]:
        """
        Extract patches, supporting cases where H or W < window_size.

        Returns:
            patches: (N, D, win_h, win_w)
            positions: [(h_start, w_start), ...] in **padded coordinate**
            original_size: (H, W)
            padded_size: (H_pad, W_pad)  <-- 新增！用于重建时知道是否需要 crop
        """
        assert tensor.ndim == 3, "Input must be (D, H, W)"
        D, H, W = tensor.shape
        win_h, win_w = self.window_size

        # 如果图像比窗口小，先 pad 到至少窗口大小
        pad_h = max(0, win_h - H)
        pad_w = max(0, win_w - W)
        H_pad = H + pad_h
        W_pad = W + pad_w

        if pad_h > 0 or pad_w > 0:
            # pad to bottom and right
            tensor_padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
        else:
            tensor_padded = tensor

        # Now extract patches from padded tensor
        ovlp_h, ovlp_w = self.win_overlap
        stride_h = win_h - ovlp_h
        stride_w = win_w - ovlp_w

        h_starts = list(range(0, H_pad - win_h + 1, stride_h))
        w_starts = list(range(0, W_pad - win_w + 1, stride_w))

        # Ensure full coverage
        if not h_starts or h_starts[-1] + win_h < H_pad:
            h_starts.append(H_pad - win_h)
        if not w_starts or w_starts[-1] + win_w < W_pad:
            w_starts.append(W_pad - win_w)

        patches = []
        positions = []
        for h in h_starts:
            for w in w_starts:
                patch = tensor_padded[:, h:h + win_h, w:w + win_w]
                patches.append(patch)
                positions.append((h, w))

        patches = torch.stack(patches, dim=0)  # (N, D, win_h, win_w)
        return patches, positions, (H, W), (H_pad, W_pad)

    def reconstruct_from_patches_3d(
            self,
            patches: torch.Tensor,
            positions: list[tuple[int, int]],
            original_size: tuple[int, int],
            padded_size: tuple[int, int],  # <-- 新增参数
            device: torch.device = None
    ) -> torch.Tensor:
        """
        Reconstruct from patches, then crop back to original size if padded.
        """
        N, D, win_h, win_w = patches.shape
        H_pad, W_pad = padded_size
        H_orig, W_orig = original_size

        if device is None:
            device = patches.device

        output = torch.zeros((D, H_pad, W_pad), device=device)
        count = torch.zeros((D, H_pad, W_pad), device=device)
        weight = torch.ones((win_h, win_w), device=device)

        for i, (h, w) in enumerate(positions):
            output[:, h:h + win_h, w:w + win_w] += patches[i] * weight
            count[:, h:h + win_h, w:w + win_w] += weight

        reconstructed_padded = output / (count + 1e-8)

        # Crop back to original size if padding was applied
        if H_pad > H_orig or W_pad > W_orig:
            reconstructed = reconstructed_padded[:, :H_orig, :W_orig]
        else:
            reconstructed = reconstructed_padded

        return reconstructed

    def get_d3_data(self, selected_slice, img_):
        # 处理 pid * D3_Z_LEN 到 (pid+1) * D3_Z_LEN 的区域
        pad_ = [0, 0]
        slice_up = [selected_slice - self.neigh_len,
                    selected_slice - self.neigh_len + self.d3_z_len]

        if slice_up[0] < 0:
            pad_[0] = -slice_up[0]
            slice_up[0] = 0
        if slice_up[1] > img_.shape[0]:
            pad_[1] = slice_up[1] - img_.shape[0]
            # full_data = torch.zeros(d3_shape, device=data.device)
        full_data = img_.unsqueeze(0)[:, slice_up[0]:slice_up[1]]
        full_data = nn.functional.pad(full_data,
                                      pad=(0, 0, 0, 0, pad_[0], pad_[1]),
                                      mode='constant', value=0)
        x = full_data[:, D3_Z_LEN // 2:D3_Z_LEN // 2 + 1]
        d3_data = {
            'full_data': full_data
        }

        print(f'gt_prompt: {selected_slice - 1}')
        return x, d3_data

    def pred_w_d3_data(self, d3_data, do_mirror):
        """
        :param d3_data:
        :param do_mirror: 是否使用TTA加强
        :return:
        """
        with torch.no_grad():
            network = self.MODEL.eval()
            full_data = d3_data['full_data']
            full_seg = torch.zeros_like(full_data)
            if 'full_seg' in d3_data:
                full_seg = d3_data['full_seg']
            need_sl = False
            d3_data_list = [d3_data]
            if full_data.shape[2] != self.window_size[0] or full_data.shape[3] != self.window_size[1]:
                print('need sl win in x, y')
                need_sl = True
            if need_sl:
                full_d_p, pos, orig_size, pad_sz = self.extract_patches_3d(full_data.squeeze(0))
                full_s_p, s_pos, s_orig_size, s_pad_sz = self.extract_patches_3d(full_seg.squeeze(0))
                batch_ori = full_s_p.shape[0]
                d3_data_list = []
                for i in range(math.ceil(batch_ori / self.max_batch)):
                    d3_data_list.append(
                        {
                            'full_data': full_d_p[i * self.max_batch:(i + 1) * self.max_batch],
                            'full_seg': full_s_p[i * self.max_batch:(i + 1) * self.max_batch]
                        }
                    )
            outs = []
            for d3_data in d3_data_list:
                if hasattr(network, 'neigh_prompt'):
                    out = network(d3_data['full_data'].unsqueeze(1), d3_data=d3_data)
                else:
                    out = network(d3_data['full_data'].unsqueeze(1))

                if do_mirror:
                    # mirror TTA
                    mirror_axes = [2, 3, 4]
                    axes_combinations = [
                        c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
                    ]
                    for axes in axes_combinations:
                        flip_full = d3_data['full_data']
                        flip_full = torch.flip(flip_full.unsqueeze(1), axes).squeeze(1)
                        flip_d3_data = {
                            'full_data': flip_full
                        }
                        if 'full_seg' in d3_data:
                            flip_seg = d3_data['full_seg']
                            flip_seg = torch.flip(flip_seg.unsqueeze(1), axes).squeeze(1)
                            flip_d3_data['full_seg'] = flip_seg
                        if hasattr(network, 'neigh_prompt'):
                            out += torch.flip(network(flip_full.unsqueeze(1), d3_data=flip_d3_data), axes)
                        else:
                            out += torch.flip(network(flip_d3_data['full_data'].unsqueeze(1)), axes)
                    out /= (len(axes_combinations) + 1)
                outs.append(out)

            if need_sl:
                out = torch.concatenate(outs, dim=0)
                # 还原
                out = logit_tran(out).float()
                out = self.reconstruct_from_patches_3d(
                    out.squeeze(1),  # ensure (N, 16, 320, 320)
                    positions=pos,
                    original_size=orig_size,
                    padded_size=pad_sz
                )
                out = out.unsqueeze(0).unsqueeze(0)
                out[out > 0] = 1
                out[out < 1] = 0
                out = one_hot_tran(out.long())
            else:
                out = outs[0]
            return out

    def get_center_slices_with_overlap(self, total_depth):
        """
        返回所有滑动窗口的中心 slice 索引（z 坐标）

        Args:
            total_depth (int): 图像总深度（img_.shape[0]）

        Returns:
            List[int]: 所有有效中心 slice 的 z 坐标
        """
        overlap = self.overlap
        window_size = self.d3_z_len
        if overlap >= window_size:
            raise ValueError("overlap must be < window_size")

        stride = window_size - overlap
        centers = []

        start = 0
        while start + window_size <= total_depth:
            center = start + window_size // 2
            centers.append(center)
            start += stride

        # 处理尾部不足一个 window 的情况（补全到最后）
        last_center = total_depth - window_size // 2
        if centers and last_center > centers[-1]:
            centers.append(last_center)
        elif not centers:  # 整个图像比 window 小
            centers.append(total_depth // 2)

        return centers

    def change_device(self, device):
        if device == 'gpu':
            if torch.cuda.is_available():
                self.device = 'gpu'
                if self.MODEL:
                    self.MODEL.to(device=self.device)
                if self.img:
                    self.img.to(device=self.device)
            else:
                return False
        elif device == 'npu':
            pass
            return False
        else:
            self.device = 'cpu'
            if self.MODEL:
                self.MODEL.to(device=self.device)
            if self.img:
                self.img.to(device=self.device)

        return True

    def load_model(self, path=None):
        if path is None:
            path = './checkpoints/VENLitem136_2_2211823_2432390'
        model_id = 0
        if os.path.exists(path + "/config.json"):
            with open(path + "/config.json", 'r', encoding='utf-8') as load_f:
                config = json.load(load_f)
            model_id = config["model_id"]
        # exp_name = path.split('/')[-1]
        network = get_model(input_channels=1, output_channels=2, deep_supervision=False,
                            model_id=model_id,
                            patch_size=None)
        path = path + '/fold_0/checkpoint_final.pth'
        print(f'using {path}')
        checkpoint = torch.load(path, map_location=torch.device(self.device), weights_only=False)
        new_state_dict = network.state_dict()
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in new_state_dict.keys() and key.startswith('module.'):
                key = key[7:]
            if key not in new_state_dict.keys():
                continue
            new_state_dict[key] = value

        network.load_state_dict(new_state_dict)
        network = network.eval()
        network.to(device=self.device)
        network.do_val = True
        if hasattr(network, 'full_out'):
            network.full_out = True
        self.MODEL = network

    def pred_all(self, mask):
        """
        依据mask直接滑动窗口推理全部
        :param mask:
        :return:
        """
        pred = torch.from_numpy(mask).long().to(device=self.device)
        img_ = self.img
        if self.need_re:
            img_ = self.re_img
            pred = self.resample_to_torch(pred, is_label=True, tar_shape=self.resample_shape)

        gt_ = pred.clone()

        print(f'slice len {img_.shape[0]}')
        centers = self.get_center_slices_with_overlap(img_.shape[0])
        for pid in centers:
            selected_slice = pid
            if selected_slice - self.neigh_len + self.d3_z_len > img_.shape[0]:
                selected_slice = img_.shape[0] - (self.d3_z_len // 2)
            gt_mid = gt_[selected_slice - 1:selected_slice]
            gt_mid = gt_mid.unsqueeze(0)

            x, d3_data = self.get_d3_data(selected_slice, img_)
            d3_data['full_seg'] = gt_mid
            out = self.pred_w_d3_data(d3_data, do_mirror=self.do_mirror)
            logit_out = logit_tran(out).squeeze(0).squeeze(0)

            pred[selected_slice - self.neigh_len:
                 selected_slice - self.neigh_len + self.d3_z_len] = logit_out.long()
            print(f'process : {selected_slice - self.neigh_len}-{selected_slice - self.neigh_len + self.d3_z_len}')

        if self.need_re:
            pred = self.resample_to_torch(pred, is_label=True, tar_shape=self.ori_shape)
        pred = pred.cpu().numpy()
        # TODO gui里需要tran

        return pred

    def pred(self, sl, mask):
        """
        以sl为标准，推理[sl-8, sl+7]共16层
        :param sl:
        :param mask:
        :return: np, mask
        """
        pred = torch.from_numpy(mask).long().to(device=self.device)
        img_ = self.img
        gt_ = pred[sl:sl + 1].clone()

        if self.need_re:
            print('need_re')
            gt_ = F.interpolate(
                gt_.unsqueeze(0).float(),
                size=(self.resample_shape[1], self.resample_shape[2]),
                mode='nearest'
            ).squeeze(0).long()
            img_ = self.re_img
            pred = self.resample_to_torch(pred, is_label=True, tar_shape=self.resample_shape)
            sl = round((sl + 0.5) * self.resample_shape[0] / self.ori_shape[1] - 0.5)

        # 不用clone会变！
        # print(gt_.shape)
        with torch.no_grad():
            # TODO 纠正
            selected_slice = int(sl) + 1
            pad_ = [0, 0]
            d3_z_len = D3_Z_LEN
            neigh_len = d3_z_len // 2
            if selected_slice - neigh_len + d3_z_len > img_.shape[0]:
                selected_slice = img_.shape[0] - (d3_z_len // 2)
            slice_up = [selected_slice - neigh_len,
                        selected_slice - neigh_len + d3_z_len]

            if slice_up[0] < 0:
                pad_[0] = -slice_up[0]
                slice_up[0] = 0
            if slice_up[1] > img_.shape[0]:
                pad_[1] = slice_up[1] - img_.shape[0]
                # full_data = torch.zeros(d3_shape, device=data.device)
            gt_mid = gt_.unsqueeze(0)

            full_data = img_.unsqueeze(0)[:, slice_up[0]:slice_up[1]]
            full_data = nn.functional.pad(full_data,
                                          pad=(0, 0, 0, 0, pad_[0], pad_[1]),
                                          mode='constant', value=0)
            d3_data = {
                'full_data': full_data
            }
            print(f'gt_prompt: {selected_slice - 1}')
            print(f'duel with: {slice_up[0]}-{slice_up[1]}')
            d3_data['full_seg'] = gt_mid
            out = self.pred_w_d3_data(d3_data, do_mirror=self.do_mirror)
            logit_out = logit_tran(out).squeeze(0).squeeze(0).long()
            logit_down = 0
            if pad_[0] > 0:
                logit_down = pad_[0]
            logit_up = D3_Z_LEN
            if pad_[1] > 0:
                logit_up = logit_up - pad_[1]

            logit_out = logit_out[logit_down:logit_up]

            pred[slice_up[0]:
                 slice_up[1]] = logit_out

            # 保留提示层避免修改，全0时用ai结果？
            if gt_mid.int().sum() > 0:
                print(f'using gt in {sl}')
                pred[sl:sl + 1] = gt_
            print(f'process : {selected_slice - neigh_len}-{selected_slice - neigh_len + d3_z_len}')

        if self.need_re:
            pred = self.resample_to_torch(pred, is_label=True, tar_shape=self.ori_shape)
        pred = pred.cpu().numpy()
        # if self.need_re:
        #     pred = self.resample_to_original_shape_torch(pred).cpu().numpy()
        # TODO gui里需要tran
        return pred
