#!/usr/bin/env python
import sys
import torch

"""
Usage:
  # download ViTAE from:
  https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Image-Classification

  # run the conversion, for example:
  ./convert-vitae.py ViTAEv2-S.pth.tar vitaev2_s_convert.pth

  # Then, use the weights with the following changes in config:

MODEL:
  WEIGHTS: "/path/to/vitaev2_s_convert.pth"

"""


if __name__ == "__main__":
    input = "/home/duolun/.torch/iopath_cache/detectron2/ImageNetPretrained/torchvision/ViTAEv2-S.pth.tar"
    output = "/home/duolun/.torch/iopath_cache/detectron2/ImageNetPretrained/torchvision/vitaev2_s_convert.pth"
    #

    source_weights = torch.load(input, map_location="cpu")['state_dict_ema']
    converted_weights = {}
    keys = list(source_weights.keys())

    for key in keys:
        #new_key = 'detection_transformer.backbone.0.backbone.' + key
        new_key = 'dptext_detr.backbone.0.backbone.' + key
        converted_weights[new_key] = source_weights[key]

    torch.save(converted_weights, output)