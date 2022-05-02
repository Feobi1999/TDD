# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import itertools
import numpy as np
import pickle
import operator
import json
import torch.utils.data
from detectron2.structures import BoxMode
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
# from detectron2.data.common import AspectRatioGroupedDataset, DatasetFromList, MapDataset
from detectron2.utils.file_io import PathManager
from detectron2.utils.comm import get_world_size
from detectron2.data.common import (
    DatasetFromList,
    MapDataset,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import (
    InferenceSampler,
    RepeatFactorTrainingSampler,
    TrainingSampler,
)
from detectron2.data.build import (
    trivial_batch_collator,
    worker_init_reset_seed,
    get_detection_dataset_dicts,
    build_batch_data_loader,
    filter_images_with_only_crowd_annotations,
    filter_images_with_few_keypoints,
    print_instances_class_histogram
)
from ubteacher.data.common import (
    AspectRatioGroupedSemiSupDatasetTwoCrop,
)

from ubteacher.data.common import (
    AspectRatioGroupedSemiSupDatasetThreeCrop,
)

from detectron2.data.detection_utils import check_metadata_consistency

"""
This file contains the default logic to build a dataloader for training or testing.
"""


def divide_label_unlabel(
    dataset_dicts, SupPercent, random_data_seed, random_data_seed_path
):
    num_all = len(dataset_dicts)
    num_label = int(SupPercent / 100.0 * num_all)

    # read from pre-generated data seed
    # with open(random_data_seed_path) as COCO_sup_file:
    #     coco_random_idx = json.load(COCO_sup_file)
    labeled_idx = np.random.choice(range(num_all), size=num_label, replace=False)
    # labeled_idx = np.array(coco_random_idx[str(SupPercent)][str(random_data_seed)])
    assert labeled_idx.shape[0] == num_label, "Number of READ_DATA is mismatched."

    label_dicts = []
    unlabel_dicts = []
    labeled_idx = set(labeled_idx)

    for i in range(len(dataset_dicts)):
        if i in labeled_idx:
            label_dicts.append(dataset_dicts[i])
        else:
            unlabel_dicts.append(dataset_dicts[i])

    return label_dicts, unlabel_dicts


# uesed by supervised-only baseline trainer
def build_detection_semisup_train_loader(cfg, mapper=None):

    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    # Divide into labeled and unlabeled sets according to supervision percentage
    label_dicts, unlabel_dicts = divide_label_unlabel(
        dataset_dicts,
        cfg.DATALOADER.SUP_PERCENT,
        cfg.DATALOADER.RANDOM_DATA_SEED,
        cfg.DATALOADER.RANDOM_DATA_SEED_PATH,
    )

    dataset = DatasetFromList(label_dicts, copy=False)

    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))

    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = (
            RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                label_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
            )
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    # list num of labeled and unlabeled
    logger.info("Number of training samples " + str(len(dataset)))
    logger.info("Supervision percentage " + str(cfg.DATALOADER.SUP_PERCENT))

    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


# uesed by evaluation
def build_detection_test_loader(cfg, dataset_name, mapper=None):
    dataset_dicts = get_detection_dataset_dicts_test(
        [dataset_name],
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[
                list(cfg.DATASETS.TEST).index(dataset_name)
            ]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    import pdb

    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    dataset = MapDataset(dataset, mapper)
    # pdb.set_trace()
    sampler = InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


# uesed by unbiased teacher trainer
def build_detection_semisup_train_loader_two_crops(cfg, mapper=None):
    if cfg.DATASETS.CROSS_DATASET:  # cross-dataset (e.g., coco-additional)
        label_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN_LABEL,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )



        unlabel_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN_UNLABEL,
            filter_empty=False,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )








    else:  # different degree of supervision (e.g., COCO-supervision)
        dataset_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )

        # Divide into labeled and unlabeled sets according to supervision percentage
        label_dicts, unlabel_dicts = divide_label_unlabel(
            dataset_dicts,
            cfg.DATALOADER.SUP_PERCENT,
            cfg.DATALOADER.RANDOM_DATA_SEED,
            cfg.DATALOADER.RANDOM_DATA_SEED_PATH,
        )

    label_dataset = DatasetFromList(label_dicts, copy=False)
    # exclude the labeled set from unlabeled dataset
    unlabel_dataset = DatasetFromList(unlabel_dicts, copy=False)
    # include the labeled set in unlabel dataset
    # unlabel_dataset = DatasetFromList(dataset_dicts, copy=False)

    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    label_dataset = MapDataset(label_dataset, mapper)
    unlabel_dataset = MapDataset(unlabel_dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    if sampler_name == "TrainingSampler":
        label_sampler = TrainingSampler(len(label_dataset))
        unlabel_sampler = TrainingSampler(len(unlabel_dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        raise NotImplementedError("{} not yet supported.".format(sampler_name))
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    return build_semisup_batch_data_loader_two_crop(
        (label_dataset, unlabel_dataset),
        (label_sampler, unlabel_sampler),
        cfg.SOLVER.IMG_PER_BATCH_LABEL,
        cfg.SOLVER.IMG_PER_BATCH_UNLABEL,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


def build_detection_semisup_train_loader_three_crops(cfg, mapper=None):
    if cfg.DATASETS.CROSS_DATASET:  # cross-dataset (e.g., coco-additional)
        label_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN_LABEL,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )

        gan_label_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN_LABEL_GAN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )

        unlabel_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN_UNLABEL,
            filter_empty=False,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )

    else:  # different degree of supervision (e.g., COCO-supervision)
        dataset_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )

        # Divide into labeled and unlabeled sets according to supervision percentage
        label_dicts, unlabel_dicts = divide_label_unlabel(
            dataset_dicts,
            cfg.DATALOADER.SUP_PERCENT,
            cfg.DATALOADER.RANDOM_DATA_SEED,
            cfg.DATALOADER.RANDOM_DATA_SEED_PATH,
        )

    label_dataset = DatasetFromList(label_dicts, copy=False)
    gan_label_dataset = DatasetFromList(gan_label_dicts, copy=False)
    # exclude the labeled set from unlabeled dataset
    unlabel_dataset = DatasetFromList(unlabel_dicts, copy=False)
    # include the labeled set in unlabel dataset
    # unlabel_dataset = DatasetFromList(dataset_dicts, copy=False)

    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    label_dataset = MapDataset(label_dataset, mapper)
    gan_label_dataset = MapDataset(gan_label_dataset, mapper)
    unlabel_dataset = MapDataset(unlabel_dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    if sampler_name == "TrainingSampler":
        label_sampler = TrainingSampler(len(label_dataset))
        unlabel_sampler = TrainingSampler(len(unlabel_dataset))
        gan_sampler = TrainingSampler(len(gan_label_dataset))

    elif sampler_name == "RepeatFactorTrainingSampler":
        raise NotImplementedError("{} not yet supported.".format(sampler_name))
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    return build_semisup_batch_data_loader_three_crop(
        (label_dataset, unlabel_dataset, gan_label_dataset),
        (label_sampler, unlabel_sampler, gan_sampler),

        cfg.SOLVER.IMG_PER_BATCH_LABEL,
        cfg.SOLVER.IMG_PER_BATCH_UNLABEL,
        cfg.SOLVER.IMG_PER_BATCH_GAN,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )

# batch data loader
def build_semisup_batch_data_loader_two_crop(
    dataset,
    sampler,
    total_batch_size_label,
    total_batch_size_unlabel,
    *,
    aspect_ratio_grouping=False,
    num_workers=0
):
    world_size = get_world_size()
    assert (
        total_batch_size_label > 0 and total_batch_size_label % world_size == 0
    ), "Total label batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_label, world_size
    )

    assert (
        total_batch_size_unlabel > 0 and total_batch_size_unlabel % world_size == 0
    ), "Total unlabel batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_label, world_size
    )

    batch_size_label = total_batch_size_label // world_size
    batch_size_unlabel = total_batch_size_unlabel // world_size

    label_dataset, unlabel_dataset = dataset




    label_sampler, unlabel_sampler = sampler

    if aspect_ratio_grouping:
        label_data_loader = torch.utils.data.DataLoader(
            label_dataset,
            sampler=label_sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0
            ),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        unlabel_data_loader = torch.utils.data.DataLoader(
            unlabel_dataset,
            sampler=unlabel_sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0
            ),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        return AspectRatioGroupedSemiSupDatasetTwoCrop(
            (label_data_loader, unlabel_data_loader),
            (batch_size_label, batch_size_unlabel),
        )



    else:
        raise NotImplementedError("ASPECT_RATIO_GROUPING = False is not supported yet")



def build_semisup_batch_data_loader_three_crop(
        dataset,
        sampler,
        total_batch_size_label,
        total_batch_size_unlabel,
        total_batch_size_gan,
        *,
        aspect_ratio_grouping=False,
        num_workers=0
):
    world_size = get_world_size()
    assert (
            total_batch_size_label > 0 and total_batch_size_label % world_size == 0
    ), "Total label batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_label, world_size
    )

    assert (
            total_batch_size_unlabel > 0 and total_batch_size_unlabel % world_size == 0
    ), "Total unlabel batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_label, world_size
    )

    assert (
            total_batch_size_gan > 0 and total_batch_size_gan % world_size == 0
    ), "Total unlabel batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_gan, world_size
    )




    batch_size_label = total_batch_size_label // world_size
    batch_size_unlabel = total_batch_size_unlabel // world_size
    batch_size_gan = total_batch_size_gan // world_size

    label_dataset, unlabel_dataset, gan_dataset = dataset




    label_sampler, unlabel_sampler, gan_sampler = sampler

    if aspect_ratio_grouping:
        label_data_loader = torch.utils.data.DataLoader(
            label_dataset,
            sampler=label_sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0
            ),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        unlabel_data_loader = torch.utils.data.DataLoader(
            unlabel_dataset,
            sampler=unlabel_sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0
            ),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict

        gan_data_loader = torch.utils.data.DataLoader(
            gan_dataset,
            sampler=gan_sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0
            ),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict


        return AspectRatioGroupedSemiSupDatasetThreeCrop(
            (label_data_loader, unlabel_data_loader, gan_data_loader),
            (batch_size_label, batch_size_unlabel, batch_size_gan),

        )



    else:
        raise NotImplementedError("ASPECT_RATIO_GROUPING = False is not supported yet")






def get_detection_dataset_dicts_test(names, filter_empty=True, min_keypoints=0, proposal_files=None):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        names (str or list[str]): a dataset name or a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        min_keypoints (int): filter out images with fewer keypoints than
            `min_keypoints`. Set to 0 to do nothing.
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `names`.

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(names, str):
        names = [names]
    assert len(names), names
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in names]
    for dataset_name, dicts in zip(names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    if proposal_files is not None:
        assert len(names) == len(proposal_files)
        # load precomputed proposals from proposal files
        dataset_dicts = [
            load_proposals_into_dataset(dataset_i_dicts, proposal_file)
            for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
        ]

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    has_instances = "annotations" in dataset_dicts[0]
    if filter_empty and has_instances:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)
    if min_keypoints > 0 and has_instances:
        dataset_dicts = filter_images_with_few_keypoints(dataset_dicts, min_keypoints)

    if has_instances:
        try:
            class_names = MetadataCatalog.get(names[0]).thing_classes
            check_metadata_consistency("thing_classes", names)
            print_instances_class_histogram(dataset_dicts, class_names)
        except AttributeError:  # class names are not available for this dataset
            pass

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(names))
    return dataset_dicts

def load_proposals_into_dataset(dataset_dicts, proposal_file):
    """
    Load precomputed object proposals into the dataset.

    The proposal file should be a pickled dict with the following keys:

    - "ids": list[int] or list[str], the image ids
    - "boxes": list[np.ndarray], each is an Nx4 array of boxes corresponding to the image id
    - "objectness_logits": list[np.ndarray], each is an N sized array of objectness scores
      corresponding to the boxes.
    - "bbox_mode": the BoxMode of the boxes array. Defaults to ``BoxMode.XYXY_ABS``.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.
        proposal_file (str): file path of pre-computed proposals, in pkl format.

    Returns:
        list[dict]: the same format as dataset_dicts, but added proposal field.
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading proposals from: {}".format(proposal_file))

    with PathManager.open(proposal_file, "rb") as f:
        proposals = pickle.load(f, encoding="latin1")

    # Rename the key names in D1 proposal files
    rename_keys = {"indexes": "ids", "scores": "objectness_logits"}
    for key in rename_keys:
        if key in proposals:
            proposals[rename_keys[key]] = proposals.pop(key)
    import pdb
    # pdb.set_trace()
    # Fetch the indexes of all proposals that are in the dataset
    # Convert image_id to str since they could be int.
    img_ids = set({str(record["image_id"]) for record in dataset_dicts})
    id_to_index = {str(id): i for i, id in enumerate(proposals["ids"]) if str(id) in img_ids}
    # pdb.set_trace()
    # Assuming default bbox_mode of precomputed proposals are 'XYXY_ABS'
    bbox_mode = BoxMode(proposals["bbox_mode"]) if "bbox_mode" in proposals else BoxMode.XYXY_ABS

    for record in dataset_dicts:
        # Get the index of the proposal
        i = id_to_index[str(record["image_id"])]

        boxes = proposals["boxes"][i]
        objectness_logits = proposals["objectness_logits"][i]
        # Sort the proposals in descending order of the scores
        # inds = objectness_logits.argsort()[::-1]
        record["proposal_boxes"] = boxes
        record["proposal_objectness_logits"] = objectness_logits
        record["proposal_bbox_mode"] = bbox_mode
    # pdb.set_trace()
    return dataset_dicts