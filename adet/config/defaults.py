from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN


# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_C.MODEL.MOBILENET = False
_C.MODEL.BACKBONE.ANTI_ALIAS = False
_C.MODEL.RESNETS.DEFORM_INTERVAL = 1
_C.INPUT.HFLIP_TRAIN = False
_C.INPUT.CROP.CROP_INSTANCE = True


# The options for BoxInst, which can train the instance segmentation model with box annotations only
# Please refer to the paper https://arxiv.org/abs/2012.02310
_C.MODEL.BOXINST = CN()
# Whether to enable BoxInst
_C.MODEL.BOXINST.ENABLED = False
_C.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED = 10

_C.MODEL.BOXINST.PAIRWISE = CN()
_C.MODEL.BOXINST.PAIRWISE.SIZE = 3
_C.MODEL.BOXINST.PAIRWISE.DILATION = 2
_C.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS = 10000
_C.MODEL.BOXINST.PAIRWISE.COLOR_THRESH = 0.3



# ---------------------------------------------------------------------------- #
# TOP Module Options
# ---------------------------------------------------------------------------- #
_C.MODEL.TOP_MODULE = CN()
_C.MODEL.TOP_MODULE.NAME = "conv"
_C.MODEL.TOP_MODULE.DIM = 16

# ---------------------------------------------------------------------------- #
# Basis Module Options
# ---------------------------------------------------------------------------- #
_C.MODEL.BASIS_MODULE = CN()
_C.MODEL.BASIS_MODULE.NAME = "ProtoNet"
_C.MODEL.BASIS_MODULE.NUM_BASES = 4
_C.MODEL.BASIS_MODULE.LOSS_ON = False
_C.MODEL.BASIS_MODULE.ANN_SET = "coco"
_C.MODEL.BASIS_MODULE.CONVS_DIM = 128
_C.MODEL.BASIS_MODULE.IN_FEATURES = ["p3", "p4", "p5"]
_C.MODEL.BASIS_MODULE.NORM = "SyncBN"
_C.MODEL.BASIS_MODULE.NUM_CONVS = 3
_C.MODEL.BASIS_MODULE.COMMON_STRIDE = 8
_C.MODEL.BASIS_MODULE.NUM_CLASSES = 80
_C.MODEL.BASIS_MODULE.LOSS_WEIGHT = 0.3

# ---------------------------------------------------------------------------- #
# ViTAE-v2 Options
# ---------------------------------------------------------------------------- #
_C.MODEL.ViTAEv2 = CN()
_C.MODEL.ViTAEv2.TYPE = 'vitaev2_s'
_C.MODEL.ViTAEv2.DROP_PATH_RATE = 0.2


# ---------------------------------------------------------------------------- #
# SwinTransformer Options
# ---------------------------------------------------------------------------- #
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.TYPE = 'tiny'
_C.MODEL.SWIN.DROP_PATH_RATE = 0.2



# ---------------------------------------------------------------------------- #
# (Deformable) Transformer Options
# ---------------------------------------------------------------------------- #
_C.MODEL.TRANSFORMER = CN()
_C.MODEL.TRANSFORMER.USE_POLYGON = False
_C.MODEL.TRANSFORMER.ENABLED = True
_C.MODEL.TRANSFORMER.INFERENCE_TH_TEST = 0.3
_C.MODEL.TRANSFORMER.VOC_SIZE = 96
_C.MODEL.TRANSFORMER.NUM_CHARS = 25
_C.MODEL.TRANSFORMER.AUX_LOSS = True
_C.MODEL.TRANSFORMER.ENC_LAYERS = 6
_C.MODEL.TRANSFORMER.DEC_LAYERS = 6
_C.MODEL.TRANSFORMER.DIM_FEEDFORWARD = 1024
_C.MODEL.TRANSFORMER.HIDDEN_DIM = 256
_C.MODEL.TRANSFORMER.DROPOUT = 0.1
_C.MODEL.TRANSFORMER.NHEADS = 8
_C.MODEL.TRANSFORMER.NUM_QUERIES = 300
_C.MODEL.TRANSFORMER.ENC_N_POINTS = 4
_C.MODEL.TRANSFORMER.DEC_N_POINTS = 4
_C.MODEL.TRANSFORMER.POSITION_EMBEDDING_SCALE = 6.283185307179586  # 2 PI
_C.MODEL.TRANSFORMER.NUM_FEATURE_LEVELS = 4
_C.MODEL.TRANSFORMER.NUM_CTRL_POINTS = 16
_C.MODEL.TRANSFORMER.MIN_AREA = 16
_C.MODEL.TRANSFORMER.EPQM = False # for DPText-DETR
_C.MODEL.TRANSFORMER.EFSA = False
_C.MODEL.TRANSFORMER.STRIDES = [8, 16, 32]
_C.MODEL.TRANSFORMER.NUM_CHANNELS = [512, 1024, 2048]
"""
strides = [8, 16, 32]
num_channels = [512, 1024, 2048]
"""

_C.MODEL.TRANSFORMER.LOSS = CN()
_C.MODEL.TRANSFORMER.LOSS.AUX_LOSS = True
_C.MODEL.TRANSFORMER.LOSS.OKS_LOSS = True

_C.MODEL.TRANSFORMER.LOSS.POINT_CLASS_WEIGHT = 2.0
_C.MODEL.TRANSFORMER.LOSS.POINT_COORD_WEIGHT = 5.0
_C.MODEL.TRANSFORMER.LOSS.POINT_GIOU_WEIGHT = 1.0
_C.MODEL.TRANSFORMER.LOSS.POINT_OKS_WEIGHT = 0.3

_C.MODEL.TRANSFORMER.LOSS.BOX_CLASS_WEIGHT = 2.0
_C.MODEL.TRANSFORMER.LOSS.BOX_COORD_WEIGHT = 10.0
_C.MODEL.TRANSFORMER.LOSS.BOX_GIOU_WEIGHT = 2.0

_C.MODEL.TRANSFORMER.LOSS.FOCAL_ALPHA = 0.25
_C.MODEL.TRANSFORMER.LOSS.FOCAL_GAMMA = 2.0
#_C.MODEL.TRANSFORMER.LOSS.POINT_BOX_INDEX = [[0, 7, 8, 15], [0, 4, 11, 15], [1, 5, 10, 14], [2, 6, 9, 13], [3, 7, 8, 12]]
#_C.MODEL.TRANSFORMER.LOSS.BOX_INDEX = [0, 1, 2, 1, 2, 3, 2, 3, 0, 0, 1, 3]
_C.MODEL.TRANSFORMER.LOSS.POINT_BOX_INDEX = None
_C.MODEL.TRANSFORMER.LOSS.BOX_INDEX = None


_C.SOLVER.OPTIMIZER = "ADAMW"
_C.SOLVER.LR_BACKBONE = 1e-5
_C.SOLVER.LR_BACKBONE_NAMES = []
_C.SOLVER.LR_LINEAR_PROJ_NAMES = []
_C.SOLVER.LR_LINEAR_PROJ_MULT = 0.1


_C.TEST.DET_ONLY = True
_C.TEST.USE_LEXICON = False
# 1 - Full lexicon (for totaltext, ctw1500...)
_C.TEST.LEXICON_TYPE = 1
_C.TEST.WEIGHTED_EDIT_DIST = False
