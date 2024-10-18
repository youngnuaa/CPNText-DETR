import os

from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

from .datasets.text import register_text_instances

# register plane reconstruction

_PREDEFINED_SPLITS_PIC = {
    "pic_person_train": ("pic/image/train", "pic/annotations/train_person.json"),
    "pic_person_val": ("pic/image/val", "pic/annotations/val_person.json"),
}

metadata_pic = {
    "thing_classes": ["person"]
}

_PREDEFINED_SPLITS_TEXT = {
    # training sets with polygon annotations
        "syntext1_poly_train_pos": (
        "/home/data3/data_wy/dept_ocr_data/data/datasets/syntext1/train_poly_pos.json",
        "/home/data3/data_wy/dept_ocr_data/data/syntext1/syntext_word_eng",
    ),
        "textocr_train_pos": (
        "/home/data3/data_wy/dept_ocr_data/data/datasets/textocr/train_37voc_poly_pos_1.json",
        "/home/data3/data_wy/dept_ocr_data/data/textocr/train_images",
    ),
        "syntext2_poly_train_pos": (
        "/home/data3/data_wy/dept_ocr_data/data/datasets/syntext2/train_poly_pos.json",
        "/home/data3/data_wy/dept_ocr_data/data/syntext2/syntext2/emcs_imgs",
    ),
        "mlt_poly_train_pos": (
        "/home/data3/data_wy/dept_ocr_data/data/datasets/mlt/train_poly_pos.json",
        "/home/data3/data_wy/dept_ocr_data/data/mlt2017/MLT_train_images",
    ),
        "totaltext_poly_train_ori": (
        "/home/data3/data_wy/dept_ocr_data/data/datasets/totaltext/train_poly_ori.json",
        "/home/data3/data_wy/dept_ocr_data/data/totaltext/train_images_rotate"),

        "totaltext_poly_train_pos": (
        "/home/data3/data_wy/dept_ocr_data/data/datasets/totaltext/train_poly_pos.json",
        "/home/data3/data_wy/dept_ocr_data/data/totaltext/train_images_rotate",
    ),
        "totaltext_poly_train_rotate_ori": (
        "/home/data3/data_wy/dept_ocr_data/data/datasets/totaltext/train_poly_rotate_ori.json",
        "/home/data3/data_wy/dept_ocr_data/data/totaltext/train_images_rotate"),

        "totaltext_poly_train_rotate_pos": (
        "/home/data3/data_wy/dept_ocr_data/data/datasets/totaltext/train_poly_rotate_pos.json",
        "/home/data3/data_wy/dept_ocr_data/data/totaltext/train_images_rotate",
    ),
        "ctw1500_poly_train_rotate_pos": (
        "/home/data3/data_wy/dept_ocr_data/data/datasets/ctw1500/train_poly_rotate_pos.json",
        "/home/data3/data_wy/dept_ocr_data/data/ctw1500/train_images_rotate",
    ),

    "lsvt_poly_train_pos": (
        "/home/data3/data_wy/dept_ocr_data/data/datasets/lsvt/train_poly_pos.json",
        "/home/data3/data_wy/dept_ocr_data/data/LSVT/rename_lsvtimg_train",
    ),

    "art_poly_train_pos": (
        "/home/data3/data_wy/dept_ocr_data/data/datasets/art/train_poly_pos.json",
        "/home/data3/data_wy/dept_ocr_data/data/art/train_images_rotate",
    ),

    "art_poly_train_rotate_pos": (
        "/home/data3/data_wy/dept_ocr_data/data/datasets/art/train_poly_rotate_pos.json",
        "/home/data3/data_wy/dept_ocr_data/data/art/train_images_rotate",
    ),

    "art_test": (
        "/home/data3/data_wy/dept_ocr_data/data/datasets/art/test_poly.json",
        "/home/data3/data_wy/dept_ocr_data/data/art/test_images",
    ),

    #"lsvt_poly_train_pos": ("lsvt/train_images","lsvt/train_poly_pos.json"),
    #"art_poly_train_pos": ("art/train_images_rotate","art/train_poly_pos.json"),
    #"art_poly_train_rotate_pos": ("art/train_images_rotate","art/train_poly_rotate_pos.json"),
    #-------------------------------------------------------------------------------------------------------
    "totaltext_poly_test": (
    "/home/data3/data_wy/dept_ocr_data/data/datasets/totaltext/test_poly.json",
    "/home/data3/data_wy/dept_ocr_data/data/totaltext/test_images_rotate"
                            ),
    "totaltext_poly_test_rotate": (
    "/home/data3/data_wy/dept_ocr_data/data/datasets/totaltext/test_poly_rotate.json",
    "/home/data3/data_wy/dept_ocr_data/data/totaltext/test_images_rotate"
                                   ),
    "ctw1500_poly_test": (
    "/home/data3/data_wy/dept_ocr_data/data/datasets/ctw1500/test_poly.json",
    "/home/data3/data_wy/dept_ocr_data/data/ctw1500/test_images"
                          ),
    #"art_test": ("art/test_images","art/test_poly.json"),
    #"inversetext_test": ("inversetext/test_images","inversetext/test_poly.json"),
}

metadata_text = {
    "thing_classes": ["text"]
}


def register_all_coco(root="datasets"):
    for key, (json_file, image_root) in _PREDEFINED_SPLITS_TEXT.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_text_instances(
            key,
            metadata_text,
            json_file,
            image_root,
        )


register_all_coco()
