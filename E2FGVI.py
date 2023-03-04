import mindspore
import mindspore.nn as nn

from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmcv.cnn import constant_init

from model.modules.flow_comp import flow_warp

class BidirectionalPropagation(nn.Module):
    def __init__(self, channel):
        super(BidirectionalPropagation, self).__init__()
        modules = ['backward_', 'forward_']
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.channel = channel

        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderAlignment(
                2 * channel, channel, 3, padding=1, deform_groups=16)

            self.backbone[module] = nn.SequentialCell(
                nn.Conv2d((2 + i) * channel, channel, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(channel, channel, 3, 1, 1),
            )

        self.fusion = nn.Conv2d(2 * channel, channel, 1, 1, 0)

    def forward(self, x, flows_backward, flows_forward):
        """
        x shape : [b, t, c, h, w]
        return [b, t, c, h, w]
        """
        b, t, c, h, w = x.shape
        feats = {}
        feats['spatial'] = [x[:, i, :, :, :] for i in range(0, t)]

        for module_name in ['backward_', 'forward_']:

            feats[module_name] = []

            frame_idx = range(0, t)
            flow_idx = range(-1, t - 1)
            mapping_idx = list(range(0, len(feats['spatial'])))
            mapping_idx += mapping_idx[::-1]

            if 'backward' in module_name:
                frame_idx = frame_idx[::-1]
                flows = flows_backward
            else:
                flows = flows_forward

            feat_prop = mindspore.ops.Zeros((b, self.channel, h, w),x.dtype)
            for i, idx in enumerate(frame_idx):
                feat_current = feats['spatial'][mapping_idx[idx]]

                if i > 0:
                    flow_n1 = flows[:, flow_idx[i], :, :, :]
                    cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                    # initialize second-order features
                    feat_n2 = mindspore.ops.ZerosLike(feat_prop)
                    flow_n2 = mindspore.ops.ZerosLike(flow_n1)
                    cond_n2 = mindspore.ops.ZerosLike(cond_n1)
                    if i > 1:
                        feat_n2 = feats[module_name][-2]
                        flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                        flow_n2 = flow_n1 + flow_warp(
                            flow_n2, flow_n1.permute(0, 2, 3, 1))
                        cond_n2 = flow_warp(feat_n2,
                                            flow_n2.permute(0, 2, 3, 1))

                    cond =  mindspore.ops.Concat([cond_n1, feat_current, cond_n2], dim=1)
                    feat_prop = mindspore.ops.Concat([feat_prop, feat_n2], dim=1)
                    feat_prop = self.deform_align[module_name](feat_prop, cond,
                                                               flow_n1,
                                                               flow_n2)

                feat = [feat_current] + [
                    feats[k][idx]
                    for k in feats if k not in ['spatial', module_name]
                ] + [feat_prop]

                feat = mindspore.ops.Concat(feat, dim=1)
                feat_prop = feat_prop + self.backbone[module_name](feat)
                feats[module_name].append(feat_prop)

            if 'backward' in module_name:
                feats[module_name] = feats[module_name][::-1]

        outputs = []
        for i in range(0, t):
            align_feats = [feats[k].pop(0) for k in feats if k != 'spatial']
            align_feats = mindspore.ops.Concat(align_feats, dim=1)
            outputs.append(self.fusion(align_feats))

        return mindspore.ops.Stack(outputs, dim=1) + x