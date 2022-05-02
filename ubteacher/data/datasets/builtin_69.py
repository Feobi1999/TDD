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
        "Cityscape/foggy_cityscape/train",
        "Cityscape/annotations/cityscapes_foggy_train_cocostyle.json",
    ),
    "cityscape_coco_unlabel": (
        "cityscapes/leftImg8bit/train",
        "Cityscapes/annotations/cityscapes_train_cocostyle.json",
    ),

    "cityscape_caronly_unlabel": (
        "Cityscape/train2017",
        "Cityscape/annotations/cityscapes_train_caronly_cocostyle.json",
    ),

    'bdd100k_train_unlabel': (
        'BDD100K/train',
        'BDD100K/annatations/det_v2_train_release_cocostyle_daytime.json'),

    'clipart_train_unlabel': (
        'Clipart/JPEGImages',
        'Clipart/clipart_coco_style.json'),

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
_root='/data0/mzhe/dataset'
register_coco_unlabel(_root)
register_coco_instances('cityscape_coco_train', {},
                        "/data0/mzhe/dataset/Cityscape/annotations/cityscapes_train_cocostyle.json",
                       '/data0/mzhe/dataset/Cityscape/train2017')

register_coco_instances('cityscape_coco_train_align', {},
                        "/data0/mzhe/dataset/Cityscape/annotations/cityscapes_train_cocostyle_align.json",
                        '/data0/mzhe/dataset/Cityscape/train2017')







register_coco_instances('cityscape_coco_test', {},
                        "/data0/mzhe/dataset/Cityscape/annotations/cityscapes_val_cocostyle.json",
                       '/data0/mzhe/dataset/Cityscape/val2017')

register_coco_instances('cityscape_car_only_val', {},
                        '/data0/mzhe/dataset/Cityscape/annotations/cityscapes_val_caronly_cocostyle.json',
                       '/data0/mzhe/dataset/Cityscape/val')

register_coco_instances('cityscape_car_only_transfer_val', {},
                        '/media/sda2/mzhe/datasets/Cityscapes/cocoAnnotations/cityscapes_val_caronly_cocostyle.json',
                       '/media/sda2/mzhe/datasets/car_trans_image')

register_coco_instances('foggy_cityscape_coco_val', {},
                        '/data0/mzhe/dataset/Cityscape/foggy_cityscape/cityscapes_foggy_val_cocostyle.json',
                       '/data0/mzhe/dataset/Cityscape/foggy_cityscape/val2017/')

register_coco_instances('foggy_trans_cityscape_coco_val', {},
                        '/media/sda2/mzhe/datasets/Cityscapes/cocoAnnotations/cityscapes_foggy_val_cocostyle.json',
                       '/media/sda2/mzhe/datasets/trans_image')

# register_coco_instances('cityscape_coco_val', {},
#                         '/media/sda2/mzhe/datasets/Cityscapes/cocoAnnotations/cityscapes_val_cocostyle.json',
#                        '/media/sda2/mzhe/datasets/Cityscapes/val20177')

register_coco_instances('sim10k_coco_train', {},
                        '/data0/mzhe/dataset/Sim10k/car_instances.json',
                       '/data0/mzhe/dataset/Sim10k/JPEGImages')

register_coco_instances('bdd100k_coco_train', {},
                        '/data0/mzhe/dataset/BDD100K/annatations/det_v2_train_release_cocostyle_daytime.json',
                       '/data0/mzhe/dataset/BDD100K/train')

register_coco_instances('bdd100k_coco_val', {},
                        '/data0/mzhe/dataset/BDD100K/annatations/det_v2_val_release_cocostyle_daytime.json',
                       '/data0/mzhe/dataset/BDD100K/val')


register_coco_instances('kitti_coco_train', {},
                        '/data0/mzhe/dataset/KITTI/caronly_training.json',
                        '/data0/mzhe/dataset/KITTI/train')


register_coco_instances('voc_clip_0712', {},
                        '/data0/mzhe/dataset/VOC_to_Clip/VOC_dt_0712_coco_style.json',
                        '/data0/mzhe/dataset/VOC_to_Clip/VOC0712')

# register_coco_instances('voc_clip_2007', {},
#                         '/data0/mzhe/dataset/VOC_to_Clip/VOC2007/coco_style_dt_clipart_2007_trainval.json',
#                         '/data0/mzhe/dataset/VOC_to_Clip/VOC0712')



# register_coco_instances('voc_2007', {},
#                         '/data0/mzhe/dataset/VOCdevkit/VOC2007/coco_style_voc2007_trainval.json',
#                         '/data0/mzhe/dataset/VOCdevkit/VOC2007/JPEGImages')
#
# register_coco_instances('voc_2012', {},
#                         '/data0/mzhe/dataset/VOCdevkit/VOC2012/coco_style_voc2012_trainval.json',
#                         '/data0/mzhe/dataset/VOCdevkit/VOC2012/JPEGImages')


register_coco_instances('voc_0712', {},
                        '/data0/mzhe/dataset/VOC_to_Clip/VOC_dt_0712_coco_style.json',
                        '/data0/mzhe/dataset/VOCdevkit/VOC0712')



register_coco_instances('clipart_train_label',  {},
                        '/data0/mzhe/dataset/Clipart/clipart_coco_style.json',
                        '/data0/mzhe/dataset/Clipart/JPEGImages')

register_coco_instances('foggy_train_label',  {},
                        '/data0/mzhe/dataset/Cityscape/annotations/cityscapes_foggy_train_cocostyle.json',
                        '/data0/mzhe/dataset/Cityscape/foggy_cityscape/train')


register_coco_instances('cityscape_caronly_train_label',  {},
                        '/data0/mzhe/dataset/Cityscape/annotations/cityscapes_train_caronly_cocostyle.json',
                        '/data0/mzhe/dataset/Cityscape/train2017')
