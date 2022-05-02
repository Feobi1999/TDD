# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch.nn import functional as F
from torch import nn
from typing import Dict, List, Optional, Tuple, Union
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    StandardROIHeads,
)
from detectron2.config import configurable
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from ubteacher.modeling.roi_heads.fast_rcnn import FastRCNNFocaltLossOutputLayers

import numpy as np
from detectron2.modeling.poolers import ROIPooler

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ubteacher.modeling.roi_heads.roi_heads_teacher_refine import StandardROIHeadsPseudoLab_TeacherRefine
@ROI_HEADS_REGISTRY.register()
class StandardROIHeadsPseudoLab_object_relation(StandardROIHeadsPseudoLab_TeacherRefine):

    def forward(
                self,
                images: ImageList,
                features: Dict[str, torch.Tensor],
                proposals: List[Instances],
                targets: Optional[List[Instances]] = None,
                compute_loss=True,
                branch="",
                compute_val_loss=False,
        ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:

            del images
            if self.training and compute_loss:  # apply if training loss
                assert targets
                # 1000 --> 512

                proposals = self.label_and_sample_proposals(
                    proposals, targets, branch=branch
                )
            elif compute_val_loss:  # apply if val loss
                assert targets
                # 1000 --> 512
                temp_proposal_append_gt = self.proposal_append_gt
                self.proposal_append_gt = False
                proposals = self.label_and_sample_proposals(
                    proposals, targets, branch=branch
                )  # do not apply target on proposals
                self.proposal_append_gt = temp_proposal_append_gt
            # del targets

            if (self.training and compute_loss) or compute_val_loss:

                if "object_relation" in branch:
                    proposals_new, box_features = self._forward_box(
                        features, proposals, compute_loss, compute_val_loss,targets, branch
                    )
                    return proposals_new, box_features
                else:
                    losses, _ = self._forward_box(
                        features, proposals, compute_loss, compute_val_loss,targets, branch
                    )


                    if branch == "target_img_two_head_refine" or branch == "target_img_two_head_attention":
                        return losses, _
                    # pdb.set_trace()
                    return proposals, losses


                # del targets


            else:
                if "object_relation" in branch:
                    proposals_new, box_features = self._forward_box(
                        features, proposals, compute_loss, compute_val_loss, targets, branch
                    )
                    return proposals_new, box_features
                pred_instances, predictions = self._forward_box(
                    features, proposals, compute_loss, compute_val_loss,targets, branch
                )
                return pred_instances, predictions

    def _forward_box(
            self,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            compute_loss: bool = True,
            compute_val_loss: bool = False,
            targets: list=None,
            branch: str = "",
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:

        # import pdb
        # pdb.set_trace()
        #features: a list
        #[8,256,h,w],[8, 256, h, w],[8, 256, h, w],[8, 256, h, w])


        features = [features[f] for f in self.box_in_features]
        batch_size = features[0].shape[0]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])  #torch.Size([8000, 256, 7, 7])
        if "object_relation" in branch :
            return proposals, box_features


        box_features = self.box_head(box_features)   #8000,1024
        predictions = self.box_predictor(box_features)  #[8000,2],[8000,4]


        if (
                self.training and compute_loss
        ) or compute_val_loss:  # apply if training loss or val loss
            losses = self.box_predictor.losses(predictions, proposals)



            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                            proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)

            if branch == "target_img_two_head_refine" or branch == "target_img_two_head_attention":
                return proposals, predictions
            return losses, predictions


        else:
            if "weak_or_pseudo" in branch:
                #
                import pdb
                # pdb.set_trace()
                temperature = 100
                scores = predictions[0]
                softmax_score = F.softmax(scores, 1)
                A = cosinematrix(softmax_score)
                # A = torch.exp(torch.mm(box_features, box_features.t())/temperature)
                # A = torch.exp(torch.mm(scores, scores.t())/temperature)
                # A = torch.exp(torch.mm(softmax_score, softmax_score.t())/temperature)
                A = A/A.sum(1,keepdim=True)
                value, index = torch.topk(A,100)
                sim_scores=scores[index]
                sim_reg = predictions[1][index]

                alpha = 0.97
                temp = alpha*predictions[0] + (1-alpha)*torch.mm(A, predictions[0])
                # predictions = (temp , predictions[1])
                predictions_sim = (sim_scores, sim_reg)
                predictions = (temp, predictions[1])
                pred_instances, _ = self.box_predictor.inference(predictions, proposals, branch)



            elif branch == "two_head_cross_refine" or branch == "teacher_img_two_head_attention":
                return proposals, predictions

            elif branch == "two_head_both_refine":

                with torch.no_grad():

                    temperature = 100
                    scores = predictions[0]
                    softmax_score = F.softmax(scores, 1)
                    A = self.cosinematrix(softmax_score)

                    A = A/A.sum(1, keepdim=True)

                    alpha = self.label_refine_ratio
                    thresh = self.label_refine_score_thresh
                    mask_small = softmax_score < thresh
                    mask_big = softmax_score > thresh

                    temp = alpha*scores + (1-alpha)*torch.mm(A, scores)

                    temp_small = temp * mask_small
                    temp_big = scores * mask_big
                    temp_prediction = temp_big + temp_small
                    predictions = (temp_prediction, predictions[1])

                    temp = alpha*predictions[0] + (1-alpha)*torch.mm(A, predictions[0])

                    predictions = (temp, predictions[1])

                return proposals, predictions





            pred_instances, _ = self.box_predictor.inference(predictions, proposals,branch)
            del box_features
            return pred_instances, predictions


