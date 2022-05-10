#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import detectron2.utils.comm as comm
from visualizer import get_local
get_local.activate()
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
import torch
from ubteacher import add_ubteacher_config

from ubteacher.engine.source_fft_np_trainer import UBTeacherTrainer, BaselineTrainer

from ubteacher.modeling.meta_arch.cp_rcnn import Ori_TwoStagePseudoLabGeneralizedRCNN
from ubteacher.modeling.meta_arch.two_head_rcnn_refine import Two_head_TwoStagePseudoLabGeneralizedRCNN_REFINE
from ubteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from ubteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import ubteacher.data.datasets.builtin_69


from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel

from ubteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
from ubteacher.modeling.roi_heads.roi_heads_two_head_refine_relation import StandardROIHeadsPseudoLab_object_relation


from ubteacher.engine.source_fft_np_trainer_two_head_version2_object_relation import Two_head_fft_UBTeacherTrainer_V2_object_relation
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def setup(args):
    """Create configs and perform basic setups."""
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if cfg.SEMISUPNET.Trainer == "joint_pretrain":
        Trainer = UBTeacherTrainer


    #use
    elif cfg.SEMISUPNET.Trainer == "fft_ub_two_head_v2_object_relation":
        Trainer = Two_head_fft_UBTeacherTrainer_V2_object_relation

    else:
        raise ValueError("Trainer Name is not found.")


    if args.eval_only:

        if cfg.SEMISUPNET.Trainer == "joint_pretrain":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)


            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)

            res = Trainer.test(cfg, ensem_ts_model.modelStudent)


        elif "two_head" in cfg.SEMISUPNET.Trainer:
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)

            res1 = Trainer.test_two_head(cfg, ensem_ts_model.modelTeacher,head=1)

            # res3 = Trainer.test_two_head(cfg, ensem_ts_model.modelTeacher,head=2)

            return res1


        else:
            model = Trainer.build_model(cfg)

            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    # init need_copy =1  ckpt contain need_copy then cover the 1 to 0
    if args.num_gpus>1:

        if "two_head" in cfg.SEMISUPNET.Trainer and trainer.model.state_dict()["module.need_copy"]:
            roi_head_dict = trainer.model.module.roi_heads.state_dict()
            trainer.model.module.roi_heads_2.load_state_dict(roi_head_dict)
            trainer.model.module.register_buffer("need_copy", torch.zeros(1).cuda())
    else:
        if "two_head" in cfg.SEMISUPNET.Trainer and trainer.model.state_dict()["need_copy"]:
            roi_head_dict = trainer.model.roi_heads.state_dict()
            trainer.model.roi_heads_2.load_state_dict(roi_head_dict)
            trainer.model.register_buffer("need_copy", torch.zeros(1).cuda())
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
