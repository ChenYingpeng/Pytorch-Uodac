from __future__ import division

import torch
import torch.nn as nn

from mmdet import ops
from mmdet.core import force_fp32
from ..registry import ROI_EXTRACTORS
import numpy as np
import random

@ROI_EXTRACTORS.register_module
class SingleRoIExtractor(nn.Module):
    """Extract RoI features from a single level feature map.

    If there are mulitple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 add_context=False,#add by chen for add_context
                 add_roimix=False,#add by chen for add_context
                 finest_scale=56):
        super(SingleRoIExtractor, self).__init__()
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale
        self.fp16_enabled = False
        self.add_context = add_context #add by chen for add_context
        self.pool = torch.nn.AdaptiveAvgPool2d(7) #add by chen for add_context
        self.add_roimix = add_roimix #add by chen for add_roimix

    @property
    def num_inputs(self):
        """int: Input feature map levels."""
        return len(self.featmap_strides)

    def init_weights(self):
        pass

    def build_roi_layers(self, layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1] + 1) * (rois[:, 4] - rois[:, 2] + 1))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def roi_rescale(self, rois, scale_factor):
        cx = (rois[:, 1] + rois[:, 3]) * 0.5
        cy = (rois[:, 2] + rois[:, 4]) * 0.5
        w = rois[:, 3] - rois[:, 1] + 1
        h = rois[:, 4] - rois[:, 2] + 1
        new_w = w * scale_factor
        new_h = h * scale_factor
        x1 = cx - new_w * 0.5 + 0.5
        x2 = cx + new_w * 0.5 - 0.5
        y1 = cy - new_h * 0.5 + 0.5
        y2 = cy + new_h * 0.5 - 0.5
        new_rois = torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1)
        return new_rois
    
    def roi_mix(self,roi_feats):
        beta = np.random.random() / 4.0

        num = roi_feats.shape[0]
        indices = np.arange(num)
        index1,index2 = random.sample(list(indices),2)

        temp = torch.zeros_like(roi_feats[index1])
        temp = (1-beta)*roi_feats[index1] + beta*roi_feats[index2]
        roi_feats[index1] = temp

        return roi_feats

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        #context = []
        if self.add_context:  #add by chen for add_context
            context = []
            for feat in feats:
                context.append(self.pool(feat))

        out_size = self.roi_layers[0].out_size
        num_levels = len(feats)
        batch_size = feats[0].shape[0] #add by chen for add_context
        target_lvls = self.map_roi_levels(rois, num_levels)
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *out_size)
        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)
        for i in range(num_levels):
            inds = target_lvls == i
            if inds.any():
                rois_ = rois[inds, :]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                if self.add_context:  #add by chen for add_context
                    temp = torch.zeros_like(roi_feats_t)
                    for j in range(batch_size):
                        temp[rois_[:, 0] == j] = roi_feats_t[rois_[:, 0] == j] + context[i][j]
                    roi_feats_t = temp
                if self.add_roimix:
                    roi_feats_t = self.roi_mix(roi_feats_t)
                roi_feats[inds] = roi_feats_t
        return roi_feats
