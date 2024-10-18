# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
from typing import Any, Dict, List, Set
import torch
import itertools
import cv2
import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.events import EventStorage
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger
import random
from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.data.dataset_ocr_mapper import DataseOcrtMapper
from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer
from adet.evaluation import TextEvaluator, TextDetEvaluator
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
import time
import numpy as np
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    polygons_to_bitmask,
)


def synchronize():
    torch.cuda.synchronize()

class Trainer(DefaultTrainer):
    """
    This is the same Trainer except that we rewrite the
    `build_train_loader`/`resume_or_load` method.
    """

    def build_hooks(self):
        """
        Replace `DetectionCheckpointer` with `AdetCheckpointer`.

        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        """
        ret = super().build_hooks()
        for i in range(len(ret)):
            if isinstance(ret[i], hooks.PeriodicCheckpointer):
                self.checkpointer = AdetCheckpointer(
                    self.model,
                    self.cfg.OUTPUT_DIR,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
                ret[i] = hooks.PeriodicCheckpointer(self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD)
        return ret

    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger("adet.trainer")
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
            self.after_train()

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        mapper = DatasetMapperWithBasis(cfg, True)
        #mapper = DataseOcrtMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if evaluator_type == "text":
            if cfg.TEST.DET_ONLY:
                return TextDetEvaluator(dataset_name, cfg, True, output_folder)
            else:
                return TextEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("adet.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    @classmethod
    def build_optimizer(cls, cfg, model):
        def match_name_keywords(n, name_keywords):
            out = False
            for b in name_keywords:
                if b in n:
                    out = True
                    break
            return out

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY

            if match_name_keywords(key, cfg.SOLVER.LR_BACKBONE_NAMES):
                lr = cfg.SOLVER.LR_BACKBONE
            elif match_name_keywords(key, cfg.SOLVER.LR_LINEAR_PROJ_NAMES):
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.LR_LINEAR_PROJ_MULT

            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

def random_colors(all_index):
    colors = []
    for index in range(all_index):
        b = random.random()
        g = random.random()
        r = random.random()
        colors.append([int(b * 255), int(g * 255), int(r * 255)])
    colors = np.array(colors)
    return colors

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # print("defaults NUM_QUERIES", cfg.MODEL.TRANSFORMER.NUM_QUERIES)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")
    # print("after NUM_QUERIES", cfg.MODEL.TRANSFORMER.NUM_QUERIES)
    return cfg

def draw_instance_edge(mask):
    h, w = mask.shape
    l_mask = np.zeros(shape=(h+10, w+10))
    l_mask[5:5+h, 5:5+w] = mask

    l_mask = l_mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    e_img = cv2.erode(l_mask, kernel)
    l_mask = l_mask - e_img
    edge_img = l_mask[5:5+h, 5:5+w]
    return edge_img

def draw_img(masks, img, colors):
    """
    """
    mask_img = img.copy()
    for index in range(len(masks)):
        pred_mask = masks[index]
        mask = np.expand_dims(pred_mask, 2)
        edge_img = draw_instance_edge(pred_mask)
        edge_img = np.expand_dims(edge_img, 2)
        edge_img = edge_img.repeat(3, axis=2)

        mask = mask.repeat(3, axis=2)
        color = colors[index]
        rgba = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))
        rgba[..., 0][pred_mask == 1] = color[0]
        rgba[..., 1][pred_mask == 1] = color[1]
        rgba[..., 2][pred_mask == 1] = color[2]
        mask_img[mask > 0.05] = mask_img[mask > 0.05] * 0.5 + rgba[mask > 0.05] * 0.5
        mask_img[edge_img == 1] = mask_img[edge_img == 1] * 0.0 + rgba[edge_img == 1] * 0
    return mask_img


def main(args):
    cfg = setup(args)
    # print("cfg",cfg)
    colors = random_colors(300)
    device = torch.device('cuda:0')
    model = Trainer.build_model(cfg)
    model.train()
    model.to(device)
    AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False
        )

    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = False
    model.to(device)
    model.eval()

    mapper = DatasetMapperWithBasis(cfg, False)
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=mapper)
    save_show_img_path = "/home/data3/data_wy/text_show_img/art/"
    with torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            input = inputs[0]
            file_name = input["file_name"]
            #print(file_name)
            img = cv2.imread(file_name)
            img_h, img_w, _ = img.shape
            #print("img shape", img.shape)
            results = model(inputs)
            result = results[0]

            instance = result["instances"]
            polygons = instance.polygons  #n, 32
            print(polygons.shape)
            masks = []
            polygons = polygons.cpu().numpy()
            image_size = (img_h, img_w)
            for polygon in polygons:
                masks.append(polygons_to_bitmask([polygon], *image_size))

            mask_img = draw_img(masks, img, colors)
            img_name = file_name.split("/")[-1]
            save_img_path = os.path.join(save_show_img_path, img_name)
            print(save_img_path)
            cv2.imwrite(save_img_path, mask_img)
            #break

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