# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
import torch
from PIL import Image
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from adet.data.ocr_augment import *
from torchvision.transforms import ToTensor, Normalize, Compose, ColorJitter
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    polygons_to_bitmask,
)


class build_transform_gen_:
    def __init__(self, cfg, is_train: bool = True):
        self.is_train = is_train
        self.min_size_train = cfg.INPUT.MIN_SIZE_TRAIN
        self.max_size_train = cfg.INPUT.MAX_SIZE_TRAIN
        self.max_size_test = cfg.INPUT.MAX_SIZE_TEST
        self.min_size_test = cfg.INPUT.MIN_SIZE_TEST

        if self.is_train:
            self.random_crop = MyEastRandomCropData(0.5)
            self.fix_rotate = EastFixRotate()
            self.random_rotate = EastRandomRotate(30, 0.5)
            self.random_mutli_scale = RandomMultiScale(self.min_size_train, self.max_size_train)
            self.random_albu = RandomALBU(0.5)
        else:
            self.random_mutli_scale = RandomMultiScale((self.min_size_test), max_size=self.max_size_test, sample_style="choice")

    def __call__(self, ann):
        if self.is_train:
            ann = self.random_crop(ann)
            ann = self.fix_rotate(ann)
            #ann = self.random_tps(ann)
            #ann = self.random_per(ann)
            ann = self.random_rotate(ann)
            ann = self.random_mutli_scale(ann)
            ann = self.random_albu(ann)
        else:
            ann = self.random_mutli_scale(ann)
        return ann


class DataseOcrtMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """
    # @classmethod

    def __init__(self, cfg, is_train: bool = True):

        # self.augs = augs
        self.is_train = is_train
        self.image_format = cfg.INPUT.FORMAT
        self.use_instance_mask = cfg.MODEL.MASK_ON
        self.instance_mask_format = cfg.INPUT.MASK_FORMAT
        self.min_area = cfg.MODEL.TRANSFORMER.MIN_AREA

        self.build_transform_gen_ = build_transform_gen_(cfg, self.is_train)
        if is_train:
            self.tv_transform = Compose([ColorJitter(brightness=0.5)])

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        dataset_dict = self.prepare_data(dataset_dict)
        return dataset_dict

    def annotations_to_instances(self, text_polys, boxes, areas, classes, image_size):
        """
        Create an :class:`Instances` object used by the models,
        from instance annotations in the dataset dict.

        Args:
            annos (list[dict]): a list of instance annotations in one image, each
                element for one instance.
            image_size (tuple): height, width

        Returns:
            Instances:
                It will contain fields "gt_boxes", "gt_classes",
                "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
                This is the format that builtin models expect.
        """
        target = Instances(image_size)
        #if len(areas)>0:
            #return target
        target.gt_boxes = Boxes(boxes)
        target.areas = torch.as_tensor(areas, dtype=torch.float32)
        classes = torch.tensor(classes, dtype=torch.int64)
        target.gt_classes = classes
        target.polygons = torch.as_tensor(text_polys, dtype=torch.float32)
        return target

    def prepare_data(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        file_name = dataset_dict["file_name"]
        image = cv2.imread(file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        text_polys = []
        for obj in dataset_dict.pop("annotations"):
            polys = obj["polygons"]
            # polys = obj["segmentation"][0]
            polys = np.array(polys)
            polys = polys.reshape(-1, 2)
            text_polys.append(polys)


        text_polys = np.array(text_polys)

        data = {}
        data["img"] = image
        data["text_polys"] = text_polys
        # print("text polys shape", text_polys.shape)
        # print("text polys", text_polys)
        data = self.build_transform_gen_(data)

        image = data["img"]
        image_shape = image.shape[:2]


        if self.is_train:
            image = Image.fromarray(np.uint8(image))
            image = self.tv_transform(image)
            image = np.array(image)
            text_polys = data["text_polys"]  # inst_num, 16, 2

            fiter_areas = []
            fiter_boxes = []
            fiter_polys = []
            fiter_classes = []
            for poly in text_polys:
                poly_ = poly.reshape(-1)
                mask = polygons_to_bitmask([poly_], *image_shape)
                mask_sum = mask.sum()
                if mask_sum < self.min_area:
                    continue
                fiter_areas.append(mask_sum)
                fiter_polys.append(poly)
                fiter_classes.append(0)

                min_x = np.min(poly[:, 0])
                min_y = np.min(poly[:, 1])
                max_x = np.max(poly[:, 0])
                max_y = np.max(poly[:, 1])
                fiter_boxes.append([min_x, min_y, max_x, max_y])

            fiter_areas = np.array(fiter_areas)
            fiter_boxes = np.array(fiter_boxes)
            fiter_polys = np.array(fiter_polys)
            fiter_classes = np.array(fiter_classes)

            #text_polys, boxes, areas, classes,
            instances = self.annotations_to_instances(fiter_polys, fiter_boxes,
                                                      fiter_areas, fiter_classes, image_shape)
            dataset_dict["instances"] =instances


        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        return dataset_dict