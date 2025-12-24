import copy
import torch.nn.functional as F
from sp_blocks import *


class ConvBlock3D(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p=0., strike=1, kernel=3):
        super(ConvBlock3D, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=strike),
            nn.InstanceNorm3d(out_channels, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout(dropout_p),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel, padding=kernel // 2, stride=1),
            nn.InstanceNorm3d(out_channels, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock3D(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p, strike=None, kernel=3):
        super(DownBlock3D, self).__init__()
        if strike is None:
            strike = [2, 2, 2]
        self.maxpool_conv = nn.Sequential(
            ConvBlock3D(in_channels, out_channels, dropout_p, strike, kernel=kernel)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock3D(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p=0., strike=None, kernel=3):
        super(UpBlock3D, self).__init__()

        if strike is None:
            strike = [2, 2, 2]
        self.up = nn.ConvTranspose3d(
            in_channels1, in_channels2, kernel_size=strike, stride=strike)
        self.conv = ConvBlock3D(in_channels2 * 2, out_channels, dropout_p, 1, kernel=kernel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder3D(nn.Module):
    def __init__(self, params, neigh_v2=False, kernels=None):
        super(Encoder3D, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        self.stride = self.params['stride']

        self.in_conv = ConvBlock3D(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.downs = nn.ModuleList()
        for i in range(len(self.ft_chns) - 1):
            self.downs.append(
                DownBlock3D(
                    self.ft_chns[i], self.ft_chns[i + 1], self.dropout[i + 1], kernel=3,
                    strike=self.stride[i]
                )
            )

    def forward(self, x):
        x = self.in_conv(x)
        ret_x = [x.clone()]
        for i in range(len(self.downs)):
            x = self.downs[i](x)
            ret_x.append(x.clone())
        return ret_x


class Decoder3D(nn.Module):
    def __init__(self, params, deep_supervision=False, no_seg=False):
        super(Decoder3D, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        self.stride = self.params['stride']
        self.deep_supervision = deep_supervision


        self.ups = nn.ModuleList()
        for i in range(1, len(self.ft_chns)):
            self.ups.append(
                UpBlock3D(
                    self.ft_chns[-i], self.ft_chns[-(i + 1)], self.ft_chns[-(i + 1)], dropout_p=0.0, kernel=3,
                    strike=self.stride[-(i + 1)]
                )
            )

        self.no_seg = no_seg
        self.out_convs = nn.ModuleList()

        for i in range(len(self.ft_chns) - 1):
            out_seq = nn.Sequential()
            out_in_chn = self.ft_chns[i]
            kernel_size = 3
            out_seq.append(nn.Conv3d(out_in_chn, self.n_class, kernel_size=kernel_size, padding=1))
            self.out_convs.append(out_seq)

    def forward(self, feature, fea_other=None):
        output = []
        x = feature[-1]
        for i in range(len(self.ups)):
            x = self.ups[i](x, feature[-(i + 2)])
            if fea_other is not None:
                if i < len(fea_other):
                    x = x * fea_other[-(i+1)]
            output.insert(0, x.clone())

        outs = []
        for i in range(len(output)):
            outs.append(self.out_convs[i](output[i]))

        if self.no_seg:
            outs.append(output)

        if not self.deep_supervision:
            outs = outs[0]

        return outs


class UNet3D(nn.Module):
    def __init__(self, in_chns, class_num, scale=1, ds=True, feature_chns=None, strides=None,
                 no_seg=False):
        super(UNet3D, self).__init__()
        if feature_chns is None:
            feature_chns = [32, 64, 128, 256, 320]
        feature_chns = [int(x / scale) for x in feature_chns]

        if strides is None:
            strides = [2] * len(feature_chns)
        if len(feature_chns) > 5:
            strides[-2] = [1, 2, 2]

        params = {
            'in_chns': in_chns,
            'feature_chns': feature_chns,
            'dropout': [0] * len(feature_chns),
            'class_num': class_num,
            'bilinear': False,
            'acti_func': 'relu',
            'stride': strides
        }

        self.__class__.__name__ = 'UNet3D'
        self.encoder = Encoder3D(params)
        self.decoder = Decoder3D(params, deep_supervision=ds, no_seg=no_seg)
        self.ds = ds

    def set_ds(self, ds=False):
        self.ds = ds

    def get_ds(self):
        return self.ds

    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output


class UNet3dwRle(nn.Module):
    def __init__(self, in_chns, class_num=2, scale=1, ds=True, shape_=None, base_net=None,
                 do_merge=False, times=2, do_d3=False, feature_chns=None):
        """
        需要输入通道为2的3d网络
        TODO 输入一层还是多层？
        拿中间一层做提示？
        """
        super(UNet3dwRle, self).__init__()

        if shape_ is None:
            shape_ = [96, 160, 160]
        self.shape_ = shape_
        # feature_2d = [64, 128, 256, 512, 1024, 2048]
        if feature_chns is None:
            feature_chns = [32, 64, 128, 256, 320, 320]
        if base_net is None:
            base_net = UNet3D(1, class_num, no_seg=True, feature_chns=feature_chns)
        # self.first_sl_net = UNet(1, class_num, deep_supervision=False, feature_c_=None)
        self.net = base_net
        self.net.decoder.deep_supervision = True
        self.do_merge = do_merge
        first_chn = feature_chns[0]
        last_chn = first_chn
        self.rle_decoder = nn.Sequential(
            ConvBlock3D(first_chn, first_chn // 2),
            ConvBlock3D(first_chn // 2, last_chn),
            nn.Conv3d(
                in_channels=last_chn,
                out_channels=times * 2,  # 保持通道数不变
                kernel_size=(3, 1, 1),  # 深度核为3，空间核为1×1
                stride=(1, 1, 1),
                padding=(2, 0, 0),  # 深度填充2，空间不填充
                # bias=False  # 可选：是否使用偏置
            )
        )

        self.loss_type = '-ce-onehot_dice-'
        self.decoder = Empty_Decoder(ds=ds)
        self.do_d3 = do_d3
        self.__class__.__name__ = f'UNet3dwRle{self.net.__class__.__name__}'
        self.do_val = False
        self.times = times
        self.need_gt = not do_d3

    def forward(self, x, gt=None, d3_data=None):
        # full_data = x.clone()
        if not self.decoder.deep_supervision:
            self.do_val = True
        if hasattr(self.net, 'do_val'):
            self.net.do_val = not self.decoder.deep_supervision
            if self.do_val:
                # 强制
                self.net.do_val = True

        if d3_data is None:
            d3_data = {
                'full_data': x.squeeze(1)
            }
            if gt is not None:
                if isinstance(gt, list):
                    gt = gt[0]
                d3_data['full_seg'] = gt.squeeze(1)

        full_data = d3_data['full_data'] # B, Z, X, Y
        # fake_out = None
        others = {}
        other_segs = {}
        # base_index = self.shape_[0]//2
        out = self.net(full_data.unsqueeze(1))
        fea = out.pop(-1)

        others['full_out'] = out[0]
        if not self.do_val or self.do_merge:
            rle_out = self.rle_decoder(fea[0])
            others['rle_out'] = rle_out
        if self.do_merge:
            x_logit = []
            for i in range(rle_out.shape[1]):
                x_logit.append(logit_tran(rle_out[:, i]).float())
            x_logit = torch.concatenate(x_logit, dim=1)
            rle_mask = two2threeRle([x_logit, x.shape], nor=False)

            others['full_out'] = others['full_out'] + one_hot_tran(rle_mask)

        fake_out = others['full_out']

        if 'full_seg' in d3_data and not self.do_val:
            gts = d3_data['full_seg'].float()
            other_segs['full_out'] = gts.unsqueeze(1).long()

            last_gt = other_segs['full_out'].float()
            for i in range(1, len(out)):
                others[f'auto_out_ds{i}'] = out[i]
                last_gt = F.adaptive_max_pool3d(last_gt.float(), output_size=list(out[i].shape)[2:]).long()
                other_segs[f'auto_out_ds{i}'] = last_gt

            other_segs['rle_out'] = three2twoRle(gts.unsqueeze(1), times=self.times, nor=False)[0]

        if hasattr(self, 'do_d3'):
            # print(f'{self.do_d3}')
            if self.do_d3:
                fake_out = out[0]
        # others['gts'] = gts
        out = [
            fake_out,
            others,
            other_segs,
            "neigh_prompt"
        ]

        if not self.decoder.deep_supervision:
            out = out[0]
            # print(f'{self.do_d3}')
            # print(out.shape)
        return out


class Neigh3dPromptWrapv5(nn.Module):
    def __init__(self, in_chns, class_num=2, scale=1, ds=True, shape_=None, base_net=None,
                 gt_fre=5, random_type='', times=2):
        """
        用于三维
        """
        super(Neigh3dPromptWrapv5, self).__init__()

        if shape_ is None:
            shape_ = [16, 320, 320]
        self.shape_ = shape_
        if base_net is None:
            base_net = UNet3D(1, class_num, ds=False)
        # self.first_sl_net = UNet(1, class_num, deep_supervision=False, feature_c_=None)
        self.net = base_net
        self.net.decoder.deep_supervision = False
        self.decoder = Empty_Decoder(ds=ds)
        # self.do_d3 = True
        self.__class__.__name__ = f'NPW{self.net.__class__.__name__}'
        self.do_val = False
        self.neigh_prompt = True
        self.last_predict = None
        self.first_one = False
        self.count = 0
        self.gt_fre = gt_fre
        self.full_out = False
        print(f'prompt fre: {self.gt_fre}')
        self.times = times
        self.gt_rate = [0, 0.1]
        self.random_type = random_type
        # self.gate_block = AdaptiveGuideGatingUnit()
        # feature_2d = [64, 128, 256, 512, 1024, 2048]
        # self.net2d = UNet(1 + times*2, class_num, deep_supervision=False, feature_c_=feature_2d)
        self.need_gt = True
        self.d3_z_len = shape_[0]
        # stage 3?

    def net_3d_forward(self, full_data, gt=None, ds=False, d3_data=None):
        if gt is None:
            gt = torch.zeros_like(full_data.unsqueeze(1))
        # 通道还是残差？
        # x_in = torch.concatenate(
        #     [
        #         full_data.unsqueeze(1),
        #         # self.gate_block(full_data.unsqueeze(1), gt)
        #         gt
        #     ], dim=1
        # )
        x_in = full_data.unsqueeze(1) + gt
        if ds:
            self.net.decoder.deep_supervision = True
        if hasattr(self.net, 'do_d3') and self.net.do_d3 and d3_data is not None:
            new_d3_data = copy.deepcopy(d3_data)
            new_d3_data['full_data'] = x_in.squeeze(1)
            out = self.net(x_in, d3_data=new_d3_data)
        elif hasattr(self.net, 'need_gt') and self.net.need_gt and d3_data is not None:
            full_seg = None
            if 'full_seg' in d3_data:
                full_seg = d3_data['full_seg'].unsqueeze(1)
            out = self.net(x_in, gt=full_seg)
        else:
            out = self.net(x_in)  # B, 2, x,y,z
        self.net.decoder.deep_supervision = False
        return out

    def forward(self, x, gt=None, d3_data=None):
        if d3_data is None:
            d3_data = {
                'full_data': x.squeeze(1)
            }
            if gt is not None:
                if isinstance(gt, list):
                    gt = gt[0]
                d3_data['full_seg'] = gt.squeeze(1)
        # full_data = x.clone()
        if not self.decoder.deep_supervision:
            self.do_val = True
        if hasattr(self.net, 'do_val'):
            self.net.do_val = not self.decoder.deep_supervision
            if self.do_val:
                # 强制
                self.net.do_val = True

        full_data = d3_data['full_data'] # B, Z, X, Y
        # fake_out = None
        others = {}
        other_segs = {}
        base_index = self.d3_z_len//2
        if self.do_val:
            if self.decoder.deep_supervision:
                self.first_one = True
            if 'full_seg' not in d3_data:
                self.first_one = True

            if self.first_one:
                self.first_one = False
                if self.full_out:
                    # TODO 自提示
                    # 只输入中间层
                    out_index = self.d3_z_len // 2 - 1
                    out = self.net_3d_forward(full_data)
                    out = out[:, :, out_index]
                    fake_out = out
                else:
                    out = self.net_3d_forward(full_data)
                    fake_out = out
            else:
                gts = d3_data['full_seg'].float()
                if gts.shape[1] > 1:
                    gt = gts[:, base_index-1:base_index]
                else:
                    gt = gts

                if self.random_type.find('--dil') != -1:
                    gt = F.max_pool2d(gt.float(), kernel_size=3, stride=1, padding=3 // 2)

                gt = gt.unsqueeze(1).repeat(1, 1, self.d3_z_len, 1, 1)

                out = self.net_3d_forward(full_data, gt, d3_data=d3_data)
                # if self.full_out:
                fake_out = out
                # else:
                #     fake_out = out[:, :, self.d3_z_len//2]
            others['full_out'] = fake_out
        else:
            # 训练
            gts = d3_data['full_seg'].float()
            # 不使用提示的分支
            out = self.net_3d_forward(full_data, ds=True, d3_data=d3_data)
            others['full_out'] = out[0]
            other_segs['full_out'] = gts.unsqueeze(1).long()
            last_gt = other_segs['full_out'].float()
            if isinstance(out[1], torch.Tensor):
                for i in range(1, len(out)):
                    others[f'auto_out_ds{i}'] = out[i]
                    last_gt = F.adaptive_max_pool3d(last_gt.float(), output_size=list(out[i].shape)[2:]).long()
                    other_segs[f'auto_out_ds{i}'] = last_gt
            elif isinstance(out[1], dict):
                for key in out[1].keys():
                    if key != 'full_out':
                        others[key] = out[1][key]
                        if key in out[2].keys():
                            other_segs[key] = out[2][key]

            fake_out = out[0]

            if gts.shape[1] > 1:
                gt = gts[:, base_index - 1:base_index]
            else:
                gt = gts

            if self.random_type.find('--dil') != -1:
                gt = F.max_pool2d(gt.float(), kernel_size=3, stride=1, padding=3 // 2)

            gt = gt.unsqueeze(1).repeat(1, 1, self.d3_z_len, 1, 1)

            # 使用gt提示的分支
            out = self.net_3d_forward(full_data, gt, ds=True, d3_data=d3_data)
            others['gt_pt_out'] = out[0]
            other_segs['gt_pt_out'] = gts.unsqueeze(1).long()
            last_gt = other_segs['gt_pt_out']
            if isinstance(out[1], torch.Tensor):
                for i in range(1, len(out)):
                    others[f'gt_pt_out_ds{i}'] = out[i]
                    last_gt = F.adaptive_max_pool3d(last_gt.float(), output_size=list(out[i].shape)[2:]).long()
                    other_segs[f'gt_pt_out_ds{i}'] = last_gt
            elif isinstance(out[1], dict):
                for key in out[1].keys():
                    if key != 'full_out':
                        others['gt_'+key] = out[1][key]
                        if key in out[2].keys():
                            other_segs['gt_'+key] = out[2][key]

        out = [
            fake_out,
            others,
            other_segs,
            "neigh_prompt"
        ]

        if not self.decoder.deep_supervision:
            out = out[0]
        return out


