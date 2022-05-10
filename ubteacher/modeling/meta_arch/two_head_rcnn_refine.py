# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.roi_heads.box_head import build_box_head
from ubteacher.modeling.roi_heads.fast_rcnn import FastRCNNFocaltLossOutputLayers
from utils import FDA_source_to_target
from torch.nn import functional as F
import torch
import numpy as np
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.config import configurable
from ubteacher.modeling.roi_heads.fast_rcnn import FastRCNNFocaltLossOutputLayers as Boxhead
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from torch import nn
from typing import Dict, List, Optional, Tuple
from detectron2.modeling.postprocessing import detector_postprocess
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
IMG_MEAN = torch.reshape( torch.from_numpy(IMG_MEAN), (1,3,1,1)  )
# from ubteacher.modeling.roi_heads.roi_heads import
import torch.nn.init as init
import math







@META_ARCH_REGISTRY.register()
class Two_head_TwoStagePseudoLabGeneralizedRCNN_REFINE(GeneralizedRCNN):

    @configurable
    def __init__(self, roi_heads_2, cross_alpha, cross_thresh, object_relation, both_attention, share_object_relation,group,**kwargs):

        super().__init__(**kwargs)
        # super().__init__()
        self.roi_heads_2 = roi_heads_2
        self.init_head2 = False
        self.cross_alpha = cross_alpha
        self.cross_thresh = cross_thresh
        self.register_buffer("need_copy", torch.ones(1))
        self.object_relation = object_relation
        self.both_attention = both_attention
        self.share = share_object_relation
        self.group = group
        if self.both_attention:
            self.theta = nn.Linear(9, 9)
            self.phi = nn.Linear(9, 9)
            self.g = nn.Linear(9, 9)
            self.h = nn.Linear(9, 9)
            self.Wz = nn.Parameter(torch.FloatTensor(9))
            self.Wz2 = nn.Parameter(torch.FloatTensor(9))

            self.scale = nn.Parameter(torch.FloatTensor([5]), requires_grad=False)
            init.kaiming_normal_(self.theta.weight, mode='fan_out')
            init.kaiming_normal_(self.phi.weight, mode='fan_out')
            init.kaiming_normal_(self.g.weight, mode='fan_out')
            self.theta.bias.data.fill_(0)
            self.phi.bias.data.fill_(0)
            self.g.bias.data.fill_(0)
            self.Wz.data.fill_(0)
            self.Wz2.data.fill_(0)
        if self.object_relation:
            if self.share:
                representation_size = 1024
                self.embed_dim = 64
                self.groups = self.group
                self.feat_dim = representation_size
                input_size = 12544
                self.base_stage = 2
                self.advanced_stage = 0

                self.base_num = 512
                self.advanced_num = int(self.base_num * 0.2)

                Wgs, Wqs, Wks, Wvs = [], [], [], []

                for i in range(self.base_stage + self.advanced_stage + 1):
                    r_size = input_size if i == 0 else representation_size

                    if i == self.base_stage and self.advanced_stage == 0:
                        break


                    Wgs.append(Conv2d(self.embed_dim, self.groups, kernel_size=1, stride=1, padding=0))
                    Wqs.append(make_fc(self.feat_dim, self.feat_dim))
                    Wks.append(make_fc(self.feat_dim, self.feat_dim))
                    Wvs.append(Conv2d(self.feat_dim * self.groups, self.feat_dim, kernel_size=1, stride=1, padding=0, groups=self.groups))

                    for l in [Wgs[i], Wvs[i]]:
                        torch.nn.init.normal_(l.weight, std=0.01)
                        torch.nn.init.constant_(l.bias, 0)
                self.Wgs = nn.ModuleList(Wgs)
                self.Wqs = nn.ModuleList(Wqs)
                self.Wks = nn.ModuleList(Wks)
                self.Wvs = nn.ModuleList(Wvs)
            else:
                representation_size = 1024
                self.embed_dim = 64
                self.groups = 16
                self.feat_dim = representation_size
                input_size = 12544
                self.base_stage = 2
                self.advanced_stage = 0

                self.base_num = 512
                self.advanced_num = int(self.base_num * 0.2)

                Wgs,Wgs_2, Wqs, Wks, Wvs, Wqs_2, Wks_2, Wvs_2, = [], [], [], [], [], [], [],[]

                for i in range(self.base_stage + self.advanced_stage + 1):
                    r_size = input_size if i == 0 else representation_size

                    if i == self.base_stage and self.advanced_stage == 0:
                        break


                    Wgs.append(Conv2d(self.embed_dim, self.groups, kernel_size=1, stride=1, padding=0))
                    Wqs.append(make_fc(self.feat_dim, self.feat_dim))
                    Wks.append(make_fc(self.feat_dim, self.feat_dim))
                    Wvs.append(Conv2d(self.feat_dim * self.groups, self.feat_dim, kernel_size=1, stride=1, padding=0, groups=self.groups))
                    Wgs_2.append(Conv2d(self.embed_dim, self.groups, kernel_size=1, stride=1, padding=0))
                    Wqs_2.append(make_fc(self.feat_dim, self.feat_dim))
                    Wks_2.append(make_fc(self.feat_dim, self.feat_dim))
                    Wvs_2.append(Conv2d(self.feat_dim * self.groups, self.feat_dim, kernel_size=1, stride=1, padding=0, groups=self.groups))


                    for l in [Wgs[i], Wvs[i],Wgs_2[i],Wvs_2[i]]:
                            torch.nn.init.normal_(l.weight, std=0.01)
                            torch.nn.init.constant_(l.bias, 0)
                self.Wgs = nn.ModuleList(Wgs)
                self.Wqs = nn.ModuleList(Wqs)
                self.Wks = nn.ModuleList(Wks)
                self.Wvs = nn.ModuleList(Wvs)
                self.Wgs_2=nn.ModuleList(Wgs_2)
                self.Wqs_2 = nn.ModuleList(Wqs_2)
                self.Wks_2 = nn.ModuleList(Wks_2)
                self.Wvs_2 = nn.ModuleList(Wvs_2)


    @classmethod
    def from_config(cls, cfg):


        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "roi_heads_2": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "cross_thresh": cfg.CROSS_THRESH,
            "cross_alpha": cfg.CROSS_ALPHA,
            "object_relation": cfg.OBJECT_RELATION,
            "both_attention": cfg.BOTH_ATTENTION,
            "share_object_relation": cfg.SHARE,
            "group":cfg.GROUP
        }


    def extract_position_embedding(self,position_mat, feat_dim, wave_length=1000.0):
        device = position_mat.device
        # position_mat, [num_rois, num_nongt_rois, 4]
        feat_range = torch.arange(0, feat_dim / 8, device=device)
        dim_mat = torch.full((len(feat_range),), wave_length, device=device).pow(8.0 / feat_dim * feat_range)
        dim_mat = dim_mat.view(1, 1, 1, -1).expand(*position_mat.shape, -1)

        position_mat = position_mat.unsqueeze(3).expand(-1, -1, -1, dim_mat.shape[3])
        position_mat = position_mat * 100.0

        div_mat = position_mat / dim_mat
        sin_mat, cos_mat = div_mat.sin(), div_mat.cos()

        # [num_rois, num_nongt_rois, 4, feat_dim / 4]
        embedding = torch.cat([sin_mat, cos_mat], dim=3)
        # [num_rois, num_nongt_rois, feat_dim]
        embedding = embedding.reshape(embedding.shape[0], embedding.shape[1], embedding.shape[2] * embedding.shape[3])

        return embedding

    @staticmethod
    def extract_position_matrix(bbox, ref_bbox):
        xmin, ymin, xmax, ymax = torch.chunk(ref_bbox, 4, dim=1)
        bbox_width_ref = xmax - xmin + 1
        bbox_height_ref = ymax - ymin + 1
        center_x_ref = 0.5 * (xmin + xmax)
        center_y_ref = 0.5 * (ymin + ymax)

        xmin, ymin, xmax, ymax = torch.chunk(bbox, 4, dim=1)
        bbox_width = xmax - xmin + 1
        bbox_height = ymax - ymin + 1
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)

        delta_x = center_x - center_x_ref.transpose(0, 1)
        delta_x = delta_x / bbox_width
        delta_x = (delta_x.abs() + 1e-3).log()

        delta_y = center_y - center_y_ref.transpose(0, 1)
        delta_y = delta_y / bbox_height
        delta_y = (delta_y.abs() + 1e-3).log()

        delta_width = bbox_width / bbox_width_ref.transpose(0, 1)
        delta_width = delta_width.log()

        delta_height = bbox_height / bbox_height_ref.transpose(0, 1)
        delta_height = delta_height.log()

        position_matrix = torch.stack([delta_x, delta_y, delta_width, delta_height], dim=2)
        # import pdb
        # pdb.set_trace()
        return position_matrix

    def attention_module_multi_head(self, roi_feat, ref_feat, position_embedding,
                                    feat_dim=1024, dim=(1024, 1024, 1024), group=16,
                                    index=0):
        """

        :param roi_feat: [num_rois, feat_dim]
        :param ref_feat: [num_nongt_rois, feat_dim]
        :param position_embedding: [1, emb_dim, num_rois, num_nongt_rois]
        :param feat_dim: should be same as dim[2]
        :param dim: a 3-tuple of (query, key, output)
        :param group:
        :return:
        """
        dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)

        # position_embedding, [1, emb_dim, num_rois, num_nongt_rois]
        # -> position_feat_1, [1, group, num_rois, num_nongt_rois]
        position_feat_1 = F.relu(self.Wgs[index](position_embedding))
        # aff_weight, [num_rois, group, num_nongt_rois, 1]
        aff_weight = position_feat_1.permute(2, 1, 3, 0)
        # aff_weight, [num_rois, group, num_nongt_rois]
        aff_weight = aff_weight.squeeze(3)

        # multi head
        assert dim[0] == dim[1]

        q_data = self.Wqs[index](roi_feat)
        q_data_batch = q_data.reshape(-1, group, int(dim_group[0]))
        # q_data_batch, [group, num_rois, dim_group[0]]
        q_data_batch = q_data_batch.permute(1, 0, 2)

        k_data = self.Wks[index](ref_feat)
        k_data_batch = k_data.reshape(-1, group, int(dim_group[1]))
        # k_data_batch, [group, num_nongt_rois, dim_group[1]]
        k_data_batch = k_data_batch.permute(1, 0, 2)

        # v_data, [num_nongt_rois, feat_dim]
        v_data = ref_feat

        # aff, [group, num_rois, num_nongt_rois]
        aff = torch.bmm(q_data_batch, k_data_batch.transpose(1, 2))
        aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff
        # aff_scale, [num_rois, group, num_nongt_rois]
        aff_scale = aff_scale.permute(1, 0, 2)

        # weighted_aff, [num_rois, group, num_nongt_rois]
        weighted_aff = (aff_weight + 1e-6).log() + aff_scale
        aff_softmax = F.softmax(weighted_aff, dim=2)

        aff_softmax_reshape = aff_softmax.reshape(aff_softmax.shape[0] * aff_softmax.shape[1], aff_softmax.shape[2])

        # output_t, [num_rois * group, feat_dim]
        output_t = torch.matmul(aff_softmax_reshape, v_data)
        # output_t, [num_rois, group * feat_dim, 1, 1]
        output_t = output_t.reshape(-1, group * feat_dim, 1, 1)
        # linear_out, [num_rois, dim[2], 1, 1]
        linear_out = self.Wvs[index](output_t)

        output = linear_out.squeeze(3).squeeze(2)

        return output

    def attentionG_module_multi_head_2(self, roi_feat, ref_feat, position_embedding,
                                    feat_dim=1024, dim=(1024, 1024, 1024), group=16,
                                    index=0):
        """

        :param roi_feat: [num_rois, feat_dim]
        :param ref_feat: [num_nongt_rois, feat_dim]
        :param position_embedding: [1, emb_dim, num_rois, num_nongt_rois]
        :param feat_dim: should be same as dim[2]
        :param dim: a 3-tuple of (query, key, output)
        :param group:
        :return:
        """
        dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)

        # position_embedding, [1, emb_dim, num_rois, num_nongt_rois]
        # -> position_feat_1, [1, group, num_rois, num_nongt_rois]
        position_feat_1 = F.relu(self.Wgs_2[index](position_embedding))
        # aff_weight, [num_rois, group, num_nongt_rois, 1]
        aff_weight = position_feat_1.permute(2, 1, 3, 0)
        # aff_weight, [num_rois, group, num_nongt_rois]
        aff_weight = aff_weight.squeeze(3)

        # multi head
        assert dim[0] == dim[1]

        q_data = self.Wqs_2[index](roi_feat)
        q_data_batch = q_data.reshape(-1, group, int(dim_group[0]))
        # q_data_batch, [group, num_rois, dim_group[0]]
        q_data_batch = q_data_batch.permute(1, 0, 2)

        k_data = self.Wks_2[index](ref_feat)
        k_data_batch = k_data.reshape(-1, group, int(dim_group[1]))
        # k_data_batch, [group, num_nongt_rois, dim_group[1]]
        k_data_batch = k_data_batch.permute(1, 0, 2)

        # v_data, [num_nongt_rois, feat_dim]
        v_data = ref_feat

        # aff, [group, num_rois, num_nongt_rois]
        aff = torch.bmm(q_data_batch, k_data_batch.transpose(1, 2))
        aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff
        # aff_scale, [num_rois, group, num_nongt_rois]
        aff_scale = aff_scale.permute(1, 0, 2)

        # weighted_aff, [num_rois, group, num_nongt_rois]
        weighted_aff = (aff_weight + 1e-6).log() + aff_scale
        aff_softmax = F.softmax(weighted_aff, dim=2)

        aff_softmax_reshape = aff_softmax.reshape(aff_softmax.shape[0] * aff_softmax.shape[1], aff_softmax.shape[2])

        # output_t, [num_rois * group, feat_dim]
        output_t = torch.matmul(aff_softmax_reshape, v_data)
        # output_t, [num_rois, group * feat_dim, 1, 1]
        output_t = output_t.reshape(-1, group * feat_dim, 1, 1)
        # linear_out, [num_rois, dim[2], 1, 1]
        linear_out = self.Wvs_2[index](output_t)

        output = linear_out.squeeze(3).squeeze(2)

        return output

    def cal_position_embedding(self, rois1, rois2):
        # [num_rois, num_nongt_rois, 4]
        position_matrix = self.extract_position_matrix(rois1, rois2)
        # [num_rois, num_nongt_rois, 64]
        position_embedding = self.extract_position_embedding(position_matrix, feat_dim=64)
        # [64, num_rois, num_nongt_rois]
        position_embedding = position_embedding.permute(2, 0, 1)
        # [1, 64, num_rois, num_nongt_rois]
        position_embedding = position_embedding.unsqueeze(0)

        return position_embedding




    def forward(
        self, batched_inputs, branch="supervised", domain_stats=None, given_proposals=None, val_mode=False
    ):


        '''
        batched_inputs: list of inputs
        len(batched_inputs) = image_per_batch
        batched_inputs[0] dict keys() ['file_name', 'height', 'width', 'image_id', 'image'])
        '''

        #for fft try
        #


        if (not self.training) and (not val_mode):

            return self.inference(batched_inputs,branch)


        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        elif "instances_head_1" in batched_inputs[0]:
            gt_instances_1 = [x["instances_head_1"].to(self.device) for x in batched_inputs]
            gt_instances_2= [x["instances_head_2"].to(self.device) for x in batched_inputs]

        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":

            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )


            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )




            losses = {}
            losses.update(detector_losses)
            # losses.update(new_detector_loss_2)
            losses.update(proposal_losses)
            return losses, [], [], None
        if branch == "target_supervised":
            new_detector_loss_2 = {}
            new_proposal_loss_2 = {}

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads_2(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            for key in detector_losses.keys():
                new_detector_loss_2[key + "_tgt"] = detector_losses[key]
            for key in proposal_losses.keys():
                new_proposal_loss_2[key + "_tgt"] = proposal_losses[key]


            losses = {}
            losses.update(new_detector_loss_2)
            losses.update(new_proposal_loss_2)
            return losses, [], [], None
        elif branch == "unsup_data_weak_two_head":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih_1, ROI_predictions_1 = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            proposals_roih_2, ROI_predictions_2 = self.roi_heads_2(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )


            return proposals_rpn, proposals_roih_1, proposals_roih_2
        elif branch == "student_object_relation":
            # Region proposal network  for two head result refine
            proposals_rpn_1, proposal_losses_1 = self.proposal_generator(images, features, gt_instances_1)

            proposals_rpn_2, proposal_losses_2 = self.proposal_generator(images, features, gt_instances_2)


            proposals_1, box_features_1 = self.roi_heads(
                images,
                features,
                proposals_rpn_1,
                gt_instances_1,
                compute_loss=True,
                branch=branch,
            )

            proposals_2, box_features_2 = self.roi_heads_2(
                images,
                features,
                proposals_rpn_2,
                targets=gt_instances_2,
                compute_loss=True,
                branch=branch,
            )

            # import pdb
            # pdb.set_trace()
            rois_cur = proposals_1[0].proposal_boxes.tensor
            rois_ref = proposals_2[0].proposal_boxes.tensor
            position_embedding = self.cal_position_embedding(rois_cur, rois_ref)


            # the first fc layer and attention layer
            for i in range(3):

                box_features_1 = self.roi_heads.box_head[i](box_features_1)
                box_features_2 = self.roi_heads_2.box_head[i](box_features_2)
            attention = self.attention_module_multi_head(box_features_1, box_features_2, position_embedding,
                                                         feat_dim=1024, group=self.group, dim=(1024, 1024, 1024),
                                                         index=0)
            # print("hhhhh1",self.group)
            box_features_1 = box_features_1 + attention

            #the second fc layer and attention layer
            for i in range(3, 5):
                box_features_1 = self.roi_heads.box_head[i](box_features_1)
                box_features_2 = self.roi_heads_2.box_head[i](box_features_2)

            attention_2 = self.attention_module_multi_head(box_features_1, box_features_2, position_embedding,
                                                         feat_dim=1024, group=self.group, dim=(1024, 1024, 1024),
                                                         index=1)


            box_features_1 = box_features_1 + attention_2

            predictions_1 = self.roi_heads.box_predictor(box_features_1)
            predictions_2 = self.roi_heads_2.box_predictor(box_features_2)
            detector_losses_1 = self.roi_heads.box_predictor.losses(predictions_1, proposals_1)
            detector_losses_2 = self.roi_heads_2.box_predictor.losses(predictions_2, proposals_2)

            new_proposal_loss_2 = {}
            for key in proposal_losses_2.keys():
                new_proposal_loss_2[key + "_head_2"] = proposal_losses_2[key]

            new_detector_loss_2 = {}
            for key in detector_losses_2.keys():
                new_detector_loss_2[key + "_head_2"] = detector_losses_2[key]

            losses = {}
            losses_2 = {}
            losses.update(detector_losses_1)
            losses.update(proposal_losses_1)

            # losses.update(new_detector_loss_2)
            losses_2.update(new_detector_loss_2)
            losses_2.update(new_proposal_loss_2)
            return losses, losses_2, [], None
        elif branch == "teacher_object_relation":
            # Region proposal network  for two head result refine
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )


            proposals_1, box_features_1 = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )
            proposals_2, box_features_2 = self.roi_heads_2(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )



            rois_cur = proposals_1[0].proposal_boxes.tensor
            rois_ref = proposals_2[0].proposal_boxes.tensor
            position_embedding = self.cal_position_embedding(rois_cur, rois_ref)

            for i in range(3):

                box_features_1 = self.roi_heads.box_head[i](box_features_1)
                box_features_2 = self.roi_heads_2.box_head[i](box_features_2)

            attention = self.attention_module_multi_head(box_features_1, box_features_2, position_embedding,
                                                         feat_dim=1024, group=self.group, dim=(1024, 1024, 1024),
                                                         index=0)
            box_features_1 = box_features_1 + attention

            for i in range(3, 5):
                box_features_1 = self.roi_heads.box_head[i](box_features_1)
                box_features_2 = self.roi_heads_2.box_head[i](box_features_2)

            attention_2 = self.attention_module_multi_head(box_features_1, box_features_2, position_embedding,
                                                           feat_dim=1024, group=self.group, dim=(1024, 1024, 1024),
                                                           index=1)



            box_features_1 = box_features_1 + attention_2

            predictions_1 = self.roi_heads.box_predictor(box_features_1)
            predictions_2 = self.roi_heads_2.box_predictor(box_features_2)

            proposals_roih_1, _ = self.roi_heads.box_predictor.inference(predictions=predictions_1, proposals= proposals_1, branch=branch)
            proposals_roih_2, _ = self.roi_heads_2.box_predictor.inference(predictions=predictions_2, proposals= proposals_2, branch=branch)

            return proposals_rpn, proposals_roih_1, proposals_roih_2






    def inference(
            self,
            batched_inputs,
            branch,
            detected_instances = None,
            do_postprocess= True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training
        # in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            # import pdb
            # pdb.set_trace()

            results_1, box_feature_1 = self.roi_heads(images, features, proposals, None)
            results_2, box_feature_2 = self.roi_heads_2(images, features, proposals, None)



        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            # if 'tgt' in branch:
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)



        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            # return self._postprocess(results, batched_inputs, images.image_sizes)
            return self._postprocess(results_1, batched_inputs, images.image_sizes), self._postprocess(results_2, batched_inputs, images.image_sizes),box_feature_1,box_feature_2
        else:
            return results

def cosinematrix(A):
    prod = torch.mm(A, A.t())#分子
    norm = torch.norm(A,p=2,dim=1).unsqueeze(0)#分母
    cos = prod.div(torch.mm(norm.t(),norm))
    return cos

def cosine_distance(matrix1,matrix2):
    matrix1_matrix2 = torch.mm(matrix1, matrix2.t())
    norm_1 = torch.norm(matrix1,p=2,dim=1).unsqueeze(0)#分母
    norm_2 = torch.norm(matrix2,p=2,dim=1).unsqueeze(0)#分母
    cos = matrix1_matrix2.div(torch.mm(norm_1.t(),norm_2))
    return cos

def make_fc(dim_in, hidden_dim, use_gn=False):
    '''
        Caffe2 implementation uses XavierFill, which in fact
        corresponds to kaiming_uniform_ in PyTorch
    '''
    # if use_gn:
    #     fc = nn.Linear(dim_in, hidden_dim, bias=False)
    #     nn.init.kaiming_uniform_(fc.weight, a=1)
    #     return nn.Sequential(fc, group_norm(hidden_dim))
    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=1)
    nn.init.constant_(fc.bias, 0)
    return fc
#
# def group_norm(out_channels, affine=True, divisor=1):
#     out_channels = out_channels // divisor
#     dim_per_gp = cfg.MODEL.GROUP_NORM.DIM_PER_GP // divisor
#     num_groups = cfg.MODEL.GROUP_NORM.NUM_GROUPS // divisor
#     eps = cfg.MODEL.GROUP_NORM.EPSILON # default: 1e-5
#     return torch.nn.GroupNorm(
#         get_group_gn(out_channels, dim_per_gp, num_groups),
#         out_channels,
#         eps,
#         affine
#     )

class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None

class Conv2d(torch.nn.Conv2d):
    def forward(self, x):
        if x.numel() > 0:
            return super(Conv2d, self).forward(x)
        # get output shape

        output_shape = [
            (i + 2 * p - (di * (k - 1) + 1)) // d + 1
            for i, p, di, k, d in zip(
                x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
            )
        ]
        output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
