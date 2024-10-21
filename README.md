# CPNText-DETR
CPNText-DETR: Dynamic Contour Initialization and  Normalized Loss for Accurate Arbitrarily Shaped Text  Detection
## Introduction

<img src='./1.png' alt='image' style="zoom:50%;" />

 Recent transformer-based methods for arbitrarily shaped text detection have
 shown improved performance by predicting text contour points. However,
 these methods often rely on bounding box centers for text contour points
 localization, which may not accurately fall within the text area, and use L1
 loss for contour point regression, causing imbalances across scales. In this pa
per, we propose CPNText-DETR, an innovative method that addresses these
 challenges. Specifically, we introduce a mass-based dynamic point query
 modeling strategy, which dynamically initializes contour points based on the
 mass center of the bounding box, aligning them more closely with the actual
 text shape. Additionally, we present the Contour Point Normalized (CPN)
 loss, which ensures uniform weighting for contour point regression across
 texts of varying scales, thus improving regression balance. Furthermore, we
 introduce the Parallel Enhanced Factorized Self-Attention (PEFSA) mod
ule, which captures intra- and inter-textual relationships, enhancing both
 performance and inference speed. Extensive experiments demonstrate that
 CPNText-DETR achieves state-of-the-art performance, with F-measures of
 89.6%, 89.5%, and 80.1% on the Total-Text, CTW1500, and Art datasets, re
spectively. CPNText-DETR not only surpasses existing methods in detection accuracy but also offers faster inference speed, establishing new benchmarks
 across widely-used datasets.

## Usage
- ### Installation
```
conda create -n CPNText-DETR python=3.8 -y
conda activate CPNText-DETR
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python scipy timm shapely albumentations Polygon3
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
pip install setuptools==59.5.0
git clone https://github.com/ymy-k/CPNText-DETR.git
cd CPNText-DETR
python setup.py build develop
```

- ### Training

**1. Pre-train:**
To pre-train the model for Total-Text and CTW1500, the config file should be `configs/CPNText_DETR/Pretrain/R_50_poly.yaml`. For ICDAR19 ArT, please use `configs/CPNText_DETR/Pretrain_ArT/R_50_poly.yaml`. Please adjust the GPU number according to your situation.

```
python tools/train_net.py --config-file ${CONFIG_FILE} --num-gpus 4
```

**2. Fine-tune:**
With the pre-trained model, use the following command to fine-tune it on the target benchmark. The pre-trained models are also provided.  For example:

```
python tools/train_net.py --config-file configs/GIOUText_DETR/TotalText/R_50_poly.yaml --num-gpus 4
```

- ### Evaluation
```
python tools/test_net.py --config-file ${CONFIG_FILE} --eval-only MODEL.WEIGHTS ${MODEL_PATH}
```
For ICDAR19 ArT, a file named `art_submit.json` will be saved in `output/r_50_poly/art/finetune/inference/`. The json file can be directly submitted to [the official website](https://rrc.cvc.uab.es/?ch=14) for evaluation.


## Data Preparation
We use the same dataset as DPText-DETR, if you are interested, please visit the following website https://github.com/ymy-k/DPText-DETR
