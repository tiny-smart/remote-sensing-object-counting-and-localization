"""
Backbone modules
"""
from collections import OrderedDict
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from mmpose.models import builder

from .swin_utils import load_pretrained
from util.misc import NestedTensor
from .swin_transformer_v2 import SwinTransformerV2
from ..position_encoding import build_position_encoding


class FeatsFusion(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, hidden_size=256, out_size=256, out_kernel=3):
        super(FeatsFusion, self).__init__()
        self.P5_1 = nn.Conv2d(C5_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.P5_2 = nn.Conv2d(hidden_size, out_size, kernel_size=out_kernel, stride=1, padding=out_kernel // 2)

        self.P4_1 = nn.Conv2d(C4_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.Conv2d(hidden_size, out_size, kernel_size=out_kernel, stride=1, padding=out_kernel // 2)

        self.P3_1 = nn.Conv2d(C3_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(hidden_size, out_size, kernel_size=out_kernel, stride=1, padding=out_kernel // 2)

    def forward(self, inputs):
        C3, C4, C5 = inputs
        C3_shape, C4_shape, C5_shape = C3.shape[-2:], C4.shape[-2:], C5.shape[-2:]

        P5_x = self.P5_1(C5)
        P5_upsampled_x = F.interpolate(P5_x, C4_shape)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = F.interpolate(P4_x, C3_shape)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x]


class BackboneBase_swin_v2(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, name: str, return_interm_layers: bool):
        super().__init__()
        # features = list(backbone.features.children())
        # if return_interm_layers:
        #     if name == 'vgg16_bn':
        #         self.body1 = nn.Sequential(*features[:13])
        #         self.body2 = nn.Sequential(*features[13:23])
        #         self.body3 = nn.Sequential(*features[23:33])
        #         self.body4 = nn.Sequential(*features[33:43])
        # else:
        #     if name == 'vgg16_bn':
        #         self.body = nn.Sequential(*features[:44])  # 16x down-sample
        self.backbone=backbone
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers
        self.fpn =FeatsFusion(64, 128, 256, hidden_size=num_channels, out_size=num_channels, out_kernel=3)#FeatsFusion(48, 96, 192, hidden_size=num_channels, out_size=num_channels, out_kernel=3)

    def forward(self, tensor_list: NestedTensor):
        feats = []
        if self.return_interm_layers:
            xs = tensor_list.tensors
            feats=self.backbone.forward_features(xs)
            # for idx, layer in enumerate([self.body1, self.body2, self.body3, self.body4]):
            #     xs = layer(xs)
            #     feats.append(xs)

            # feature fusion
            features_fpn = self.fpn([feats[0], feats[1], feats[2]])
            features_fpn_4x = features_fpn[0]
            features_fpn_8x = features_fpn[1]

            # get tensor mask
            m = tensor_list.mask
            assert m is not None
            mask_4x = F.interpolate(m[None].float(), size=features_fpn_4x.shape[-2:]).to(torch.bool)[0]
            mask_8x = F.interpolate(m[None].float(), size=features_fpn_8x.shape[-2:]).to(torch.bool)[0]

            out: Dict[str, NestedTensor] = {}
            out['4x'] = NestedTensor(features_fpn_4x, mask_4x)
            out['8x'] = NestedTensor(features_fpn_8x, mask_8x)
        else:
            xs = self.body(tensor_list)
            out.append(xs)

        return out


class Backbone_swin(BackboneBase_swin_v2):
    """
    VGG backbone
    """

    def __init__(self, name: str, encoder_config,pretrained=False,test=False):
        if test==False:
           img_size_t=256
        else:
            img_size_t=(512,512)
        if name == 'swin_v2':
            backbone = SwinTransformerV2(type='SwinTransformerV2',
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=16,
        pretrained_window_sizes=[12, 12, 12, 6],
        drop_path_rate=0.1,
        img_size=img_size_t,multi_scale=True)
        #    backbone = SwinTransformerV2(type='SwinTransformerV2',
        #         embed_dim=96,
        #         depths=[2, 2, 6, 2],
        #         num_heads=[3, 6, 12, 24],
        #         window_size=16,
        #         drop_path_rate=0.1,
        #         img_size=img_size_t,multi_scale=True)

            load_pretrained(pretrained, backbone, logger=None)
        num_channels=256
        super().__init__(backbone,num_channels, name, True)

        num_channels=256
        super().__init__(backbone,num_channels, name, True)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: Dict[NestedTensor] = {}
        pos = {}
        for name, x in xs.items():
            out[name] = x
            # position encoding
            pos[name] = self[1](x).to(x.tensors.dtype)
        return out, pos


def build_backbone_swin(args,test=False):
    position_embedding = build_position_encoding(args)
    encoder_config=dict(
        type='SwinTransformerV2',
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=14,
        pretrained_window_sizes=[12, 12, 12, 6],
        drop_path_rate=0.1,
        img_size=224,
    )
    backbone = Backbone_swin(args.backbone, 256,args.pretrained,test)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


if __name__ == '__main__':
    Backbone_swin('swin_v2', True)
