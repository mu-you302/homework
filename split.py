import os
from random import shuffle
import numpy as np
import cv2


def get_bbox(mask):
    """
    get bbox information from mask image
    mask: np.array, shape=(H, W)
    return x_center y_center width height"""
    W, H = mask.shape
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)  # x,y - top-left
    # get the center of the bbox
    x_center = x + w / 2
    y_center = y + h / 2
    # norm x_center, y_center
    x_center /= W
    y_center /= H
    return x_center, y_center, w / W, h / H


def convert2YOLO(
    files, mode, images_dir=r"./datasets/images", labels_dir=r"./datasets/labels"
):
    """
    save image to images_dir and label to labels_dir
    mode: str, "train", "val", "test"
    """
    os.makedirs(images_dir, exist_ok=True)
    images_dir = os.path.join(images_dir, mode)
    if len(os.listdir(images_dir)) > 0:
        print("assuming dataset has been splited...")
        return
    # if exists, remove it
    # if os.path.exists(images_dir):
    #     os.system(f"rm -rf {images_dir}")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    labels_dir = os.path.join(labels_dir, mode)
    # if exists, remove it
    # if os.path.exists(labels_dir):
    #     os.system(f"rm -rf {labels_dir}")
    os.makedirs(labels_dir, exist_ok=True)

    for file in files:
        img_path = file + ".tif"
        mask_path = file + "_mask.tif"
        name = os.path.splitext(os.path.basename(file))[0]
        img_name = name + ".jpg"
        label_name = name + ".txt"
        img_save_path = os.path.join(images_dir, img_name)
        label_save_path = os.path.join(labels_dir, label_name)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # save image
        cv2.imwrite(img_save_path, img)

        # save label
        if mask.sum() == 0:
            continue
        # get bbox
        x, y, w, h = get_bbox(mask)
        with open(label_save_path, "w") as f:
            f.write(f"0 {x} {y} {w} {h}")


# walk dir and count the number of files
root = "kaggle_3m"
dirs = os.listdir(root)

file_paths = []

for dir in dirs:
    dir_name = os.path.join(root, dir)
    if os.path.isdir(dir_name):
        files = os.listdir(dir_name)
        for file in files:
            if file.endswith("mask.tif"):
                name = os.path.splitext(file)[0][:-5]
                file_paths.append(os.path.join(dir_name, name))

# split into train, val, test 7:2:1
num = len(file_paths)
train_num = int(num * 0.7)
val_num = int(num * 0.2)
test_num = num - train_num - val_num

shuffle(file_paths)
train_files = file_paths[:train_num]
val_files = file_paths[train_num : train_num + val_num]
test_files = file_paths[train_num + val_num :]

convert2YOLO(train_files, "train")
convert2YOLO(val_files, "val")
convert2YOLO(test_files, "test")
