# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_ubteacher_config(cfg):
    """
    Add config for semisupnet.
    """
    _C = cfg
    _C.TEST.VAL_LOSS = False

    _C.MODEL.RPN.UNSUP_LOSS_WEIGHT = 1.0
    _C.MODEL.RPN.LOSS = "CrossEntropy"
    _C.MODEL.ROI_HEADS.LOSS = "CrossEntropy"

    _C.SOLVER.IMG_PER_BATCH_LABEL = 1
    _C.SOLVER.IMG_PER_BATCH_UNLABEL = 1
    _C.SOLVER.FACTOR_LIST = (1,)
    _C.SOLVER.IMS_PER_BATCH=8
    _C.SOLVER.SUB_LR  = None
    _C.DATASETS.TRAIN_LABEL = ("coco_2017_train",)
    _C.DATASETS.TRAIN_UNLABEL = ("coco_2017_train",)
    _C.DATASETS.CROSS_DATASET = False
    _C.TEST.EVALUATOR = "COCOeval"

    _C.SEMISUPNET = CN()

    # Output dimension of the MLP projector after `res5` block
    _C.SEMISUPNET.MLP_DIM = 128

    # Semi-supervised training
    _C.SEMISUPNET.Trainer = "joint_pretrain"
    _C.SEMISUPNET.BBOX_THRESHOLD = 0.7
    _C.SEMISUPNET.PSEUDO_BBOX_SAMPLE = "thresholding"
    _C.SEMISUPNET.TEACHER_UPDATE_ITER = 1
    _C.SEMISUPNET.BURN_UP_STEP = 12000
    _C.SEMISUPNET.EMA_KEEP_RATE = 0.0
    _C.SEMISUPNET.UNSUP_LOSS_WEIGHT = 4.0
    _C.SEMISUPNET.SUP_LOSS_WEIGHT = 0.5
    _C.SEMISUPNET.LOSS_WEIGHT_TYPE = "standard"
    _C.SEMISUPNET.TWO_STAGE_THRESHHOLD= False
    _C.SEMISUPNET.BBOX_THRESHOLD_1 = 0.9
    _C.SEMISUPNET.BBOX_THRESHOLD_2 = 0.7
    _C.SEMISUPNET.STEP = 10000
    _C.SEMISUPNET.Teacher_Refine = False
    # dataloader
    # supervision level
    _C.DATALOADER.SUP_PERCENT = 100.0  # 5 = 5% dataset as labeled set
    _C.DATALOADER.RANDOM_DATA_SEED = 0  # random seed to read data
    _C.DATALOADER.RANDOM_DATA_SEED_PATH = "dataseed/COCO_supervision.txt"

    _C.EMAMODEL = CN()
    _C.EMAMODEL.SUP_CONSIST = True
    _C.DA_CASE= None
    _C.DA_FPN_FEATURE = ["p2", "p3", "p4", "p5"]
    _C.CONSIST_MODE = ""
    _C.ADDITIONAL_BRANCH = 0
    _C.CONSIST_ON = False
    _C.CONTRAST_ON = False
    _C.FFT_ON = False
    _C.FFT_ON_1 = False
    _C.FFT_ON_2 = False
    _C.LABEL_REFINE_THRESH = 0.7
    _C.LABEL_REFINE_RATIO = 0.97
    _C.REFINE_ITER = 2000
    _C.CROSS_THRESH = 0.7
    _C.CROSS_ALPHA = 0.97
    _C.OBJECT_RELATION = False
    _C.BOTH_ATTENTION = False
    _C.SHARE = True
    _C.GROUP = 16
    _C.SOLVER.IMG_PER_BATCH_GAN = 8
    _C.DATASETS.TRAIN_LABEL_GAN = "voc_clip_0712"