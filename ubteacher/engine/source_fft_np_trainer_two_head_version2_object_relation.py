# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import time
import logging
import torch
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
from collections import OrderedDict
from utils import FDA_source_to_target, FDA_source_to_target_np
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from ubteacher.evaluation.evaluator import inference_on_dataset, two_head_inference_on_dataset
from ubteacher.engine.original_trainer_two_head import Two_Head_UBTeacherTrainer
from ubteacher.evaluation.coco_evaluation import COCOEvaluator
from detectron2.evaluation import verify_results


# from detectron2.evaluation import verify_results
from detectron2.engine import hooks
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)

from ubteacher.data.build import (
    build_detection_semisup_train_loader,
    build_detection_test_loader,
    build_detection_semisup_train_loader_two_crops,
)
from ubteacher.data.dataset_mapper import DatasetMapperTwoCropSeparate
from detectron2.data.dataset_mapper import DatasetMapper
from ubteacher.engine.hooks import LossEvalHook
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from ubteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from ubteacher.solver.build import build_lr_scheduler

from visualizer import get_local


# Unbiased Teacher Trainer
class Two_head_fft_UBTeacherTrainer_V2_object_relation(Two_Head_UBTeacherTrainer):

    def add_label_for_two_head(self, unlabled_data, label_1,label_2):
        for unlabel_datum, lab_inst_1, lab_inst_2 in zip(unlabled_data, label_1,label_2):
            unlabel_datum["instances_head_1"] = lab_inst_1
            unlabel_datum["instances_head_2"] = lab_inst_2
        return unlabled_data

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)

        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
        import pdb
        src_label_data_k = label_data_k.copy()

        src_image= label_data_k[0]['image'].cpu().numpy() #[3,600,1200]
        tgt_image= unlabel_data_k[0]['image'].cpu().numpy()

        src_in_trg = FDA_source_to_target_np(src_image,tgt_image, L=0.1)  # src_lbl

        label_data_k[0]['image']=torch.Tensor(src_in_trg).cuda()

        data_time = time.perf_counter() - start

        # burn-in stage (supervised training with labeled data)
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:


            # input both strong and weak supervised data into model
            label_data_q.extend(label_data_k)
            if self.cfg.CONSIST_ON:
                record_dict, _, _, _ = self.model(label_data_q, branch="consist_supervised")

            else:
                record_dict, _, _, _ = self.model(label_data_q, branch="supervised")



            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values())

        else:
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                # update copy the the whole model
                self._update_teacher_model(keep_rate=0.00)

            elif (
                    self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
            ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                self._update_teacher_model(keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

            record_dict = {}
            #  generate the pseudo-label using teacher model
            # note that we do not convert to eval mode, as 1) there is no gradient computed in
            # teacher model and 2) batch norm layers are not updated as well
            with torch.no_grad():
                if self.cfg.SEMISUPNET.Teacher_Refine:
                    (
                        proposals_rpn_unsup_k,
                        proposals_roih1_unsup_k,
                        proposals_roih2_unsup_k
                    ) = self.model_teacher(unlabel_data_k, branch="teacher_object_relation")
                else:
                    (
                        proposals_rpn_unsup_k,
                        proposals_roih1_unsup_k,
                        proposals_roih2_unsup_k
                    ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak_two_head")
            cache = get_local.cache

            #  Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD


            joint_proposal_dict = {}
            joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k

            (
                pesudo_proposals_rpn_unsup_k,
                nun_pseudo_bbox_rpn,
            ) = self.process_pseudo_label(
                proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
            )
            joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k



            # Pseudo_labeling for ROI head (bbox location/objectness)
            pesudo_proposals_roih1_unsup_k, _ = self.process_pseudo_label(
                proposals_roih1_unsup_k, cur_threshold, "roih", "thresholding"
            )
            joint_proposal_dict["proposals_pseudo_roih_1"] = pesudo_proposals_roih1_unsup_k


            pesudo_proposals_roih2_unsup_k, _ = self.process_pseudo_label(
                proposals_roih2_unsup_k, cur_threshold, "roih", "thresholding"
            )
            joint_proposal_dict["proposals_pseudo_roih_2"] = pesudo_proposals_roih2_unsup_k





            #  add pseudo-label to unlabeled data
            unlabel_data_q = self.remove_label(unlabel_data_q)
            unlabel_data_k = self.remove_label(unlabel_data_k)


            unlabel_data_q_two_label = self.add_label_for_two_head(
                unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih_1"], joint_proposal_dict["proposals_pseudo_roih_1"]
            )




            # for src domain images
            all_label_data = label_data_q + src_label_data_k
            record_all_label_data, _, _, _ = self.model(
                all_label_data, branch="supervised"
            )

            record_dict.update(record_all_label_data)

            # for fft image
            record_unlabeld_data_fft_head2, _, _, _ = self.model(
                label_data_k, branch="target_supervised"
            )
            record_dict.update(record_unlabeld_data_fft_head2)


            new_record_all_unlabel_data = {}


            record_unlabeld_data_head_1, record_unlabeld_data_head_2, _, _ = self.model(
                unlabel_data_q_two_label, branch="student_object_relation"
            )


            for key in record_unlabeld_data_head_1.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = record_unlabeld_data_head_1[
                    key
                ]


            record_dict.update(new_record_all_unlabel_data)
            record_dict.update(record_unlabeld_data_head_2)

            # import pdb
            # pdb.set_trace()
            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":

                    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        loss_dict[key] = (
                                record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                        )
                    elif 'consist' in key:
                        loss_dict[key]=(
                            record_dict[key]*0.1
                        )
                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1

            losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
