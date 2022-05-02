# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch.nn import functional as F
from torch import nn
from typing import Dict, List, Optional, Tuple, Union
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)
from detectron2.config import configurable

from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    StandardROIHeads,
)
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from ubteacher.modeling.roi_heads.fast_rcnn import FastRCNNFocaltLossOutputLayers

import numpy as np
from detectron2.modeling.poolers import ROIPooler
# from ubteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab

@ROI_HEADS_REGISTRY.register()
class StandardROIHeadsPseudoLab_TeacherRefine(StandardROIHeads):

    @configurable
    def __init__(self,label_refine_score_thresh ,label_refine_ratio, **kwargs):

        super().__init__(**kwargs)
        # super().__init__()
        self.label_refine_score_thresh = label_refine_score_thresh
        self.label_refine_ratio = label_refine_ratio



    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        label_refine_score_thresh = cfg.LABEL_REFINE_THRESH
        label_refine_ratio = cfg.LABEL_REFINE_RATIO
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )
        if cfg.MODEL.ROI_HEADS.LOSS == "CrossEntropy":
            box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        elif cfg.MODEL.ROI_HEADS.LOSS == "FocalLoss":
            box_predictor = FastRCNNFocaltLossOutputLayers(cfg, box_head.output_shape)
        else:
            raise ValueError("Unknown ROI head loss.")

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
            "label_refine_score_thresh": label_refine_score_thresh,
            "label_refine_ratio": label_refine_ratio
        }


    @torch.no_grad()
    def label_and_sample_proposals(
            self, proposals: List[Instances], targets: List[Instances], branch: str = ""
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                            trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        storage = get_event_storage()
        storage.put_scalar(
            "roi_head/num_target_fg_samples_" + branch, np.mean(num_fg_samples)
        )
        storage.put_scalar(
            "roi_head/num_target_bg_samples_" + branch, np.mean(num_bg_samples)
        )

        return proposals_with_gt
    @torch.no_grad()
    def cosinematrix(self, A):
        prod = torch.mm(A, A.t())#分子
        norm = torch.norm(A,p=2,dim=1).unsqueeze(0)#分母
        cos = prod.div(torch.mm(norm.t(),norm))
        return cos