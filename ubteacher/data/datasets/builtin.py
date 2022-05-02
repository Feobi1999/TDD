# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import contextlib
from detectron2.data import DatasetCatalog, MetadataCatalog
from fvcore.common.timer import Timer
from fvcore.common.file_io import PathManager
import io
import logging
from detectron2.data.datasets import register_coco_instances

logger = logging.getLogger(__name__)

JSON_ANNOTATIONS_DIR = ""
_SPLITS_COCO_FORMAT = {}
_SPLITS_COCO_FORMAT["coco"] = {
    "foggy_coco_2017_unlabel": (
        "cityscapes/leftImg8bit_foggy/train",
        "cityscapes/annotations/cityscapes_foggy_train_cocostyle.json",
    ),
    "cityscape_coco_unlabel": (
        "cityscapes/leftImg8bit/train",
        "Cityscapes/annotations/cityscapes_train_cocostyle.json",
    ),

    "cityscape_caronly_unlabel": (
        "cityscapes/leftImg8bit/train",
        "cityscapes/annotations/cityscapes_train_caronly_cocostyle.json",
    ),

    'bdd100k_train_unlabel': (
        'bdd100k/images/100k/train',
        'bdd100k/det_v2_train_release_cocostyle_daytime.json'),
}








def register_coco_unlabel(root):
    for _, splits_per_dataset in _SPLITS_COCO_FORMAT.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            meta = {}
            register_coco_unlabel_instances(
                key, meta, os.path.join(root, json_file), os.path.join(root, image_root)
            )


def register_coco_unlabel_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root

    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_coco_unlabel_json(json_file, image_root, name)
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )
    # MetadataCatalog.get(name).thing_classes = ["1", "2" ,"3", "4","5","6","7","8"]


def load_coco_unlabel_json(
    json_file, image_root, dataset_name=None, extra_annotation_keys=None
):
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds())
        )

    id_map = None
    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())

    imgs = coco_api.loadImgs(img_ids)

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs), json_file))

    dataset_dicts = []

    for img_dict in imgs:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        dataset_dicts.append(record)

    return dataset_dicts


# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
_root='/mnt/lustre/hemengzhe/datasets'
register_coco_unlabel(_root)
register_coco_instances('cityscape_coco_train', {},
                        "/mnt/lustre/hemengzhe/datasets/cityscapes/annotations/cityscapes_train_cocostyle.json",
                       '/mnt/lustre/hemengzhe/datasets/cityscapes/leftImg8bit/train')
register_coco_instances('cityscape_coco_test', {},
                        "/mnt/lustre/hemengzhe/datasets/cityscapes/annotations/cityscapes_val_cocostyle.json",
                       '/mnt/lustre/hemengzhe/datasets/cityscapes/leftImg8bit/val')

register_coco_instances('cityscape_car_only_val', {},
                        '/mnt/lustre/hemengzhe/datasets/cityscapes/annotations/cityscapes_val_caronly_cocostyle.json',
                       '/mnt/lustre/hemengzhe/datasets/cityscapes/leftImg8bit/val/')

register_coco_instances('cityscape_car_only_transfer_val', {},
                        '/media/sda2/mzhe/datasets/Cityscapes/cocoAnnotations/cityscapes_val_caronly_cocostyle.json',
                       '/media/sda2/mzhe/datasets/car_trans_image')

register_coco_instances('foggy_cityscape_coco_val', {},
                        '/mnt/lustre/hemengzhe/datasets/cityscapes/annotations/cityscapes_foggy_val_cocostyle.json',
                       '/mnt/lustre/hemengzhe/datasets/cityscapes/leftImg8bit_foggy/val')

register_coco_instances('foggy_trans_cityscape_coco_val', {},
                        '/media/sda2/mzhe/datasets/Cityscapes/cocoAnnotations/cityscapes_foggy_val_cocostyle.json',
                       '/media/sda2/mzhe/datasets/trans_image')

# register_coco_instances('cityscape_coco_val', {},
#                         '/media/sda2/mzhe/datasets/Cityscapes/cocoAnnotations/cityscapes_val_cocostyle.json',
#                        '/media/sda2/mzhe/datasets/Cityscapes/val20177')

register_coco_instances('sim10k_coco_train', {},
                        '/mnt/lustre/hemengzhe/datasets/sim10k/car_instances.json',
                       '/mnt/lustre/hemengzhe/datasets/sim10k/JPEGImages')

# register_coco_instances('bdd100k_coco_train', {},
#                         '/media/sda2/mzhe/BDD100K/det_v2_train_release_cocostyle_daytime.json',
#                        '/mnt/lustre/hemengzhe/datasets/bdd100k/images/100k/train/')

register_coco_instances('bdd100k_coco_val', {},
                        '/mnt/lustre/hemengzhe/datasets/bdd100k/det_v2_val_release_cocostyle_daytime.json',
                       '/mnt/lustre/hemengzhe/datasets/bdd100k/images/100k/val')


register_coco_instances('kitti_coco_train', {},
                        '/mnt/lustre/hemengzhe/datasets/kitti/caronly_training.json',
                        '/mnt/lustre/hemengzhe/datasets/kitti/training/image_2')