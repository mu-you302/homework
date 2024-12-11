# LGG Segmentation Dataset

This dataset contains brain MR images together with manual FLAIR abnormality segmentation masks.
The images were obtained from The Cancer Imaging Archive (TCIA).
They correspond to 110 patients included in The Cancer Genome Atlas (TCGA) lower-grade glioma collection with at least fluid-attenuated inversion recovery (FLAIR) sequence and genomic cluster data available.
Tumor genomic clusters and patient data is provided in `data.csv` file.


All images are provided in `.tif` format with 3 channels per image.
For 101 cases, 3 sequences are available, i.e. pre-contrast, FLAIR, post-contrast (in this order of channels).
For 9 cases, post-contrast sequence is missing and for 6 cases, pre-contrast sequence is missing.
Missing sequences are replaced with FLAIR sequence to make all images 3-channel.
Masks are binary, 1-channel images.
They segment FLAIR abnormality present in the FLAIR sequence (available for all cases).


The dataset is organized into 110 folders named after case ID that contains information about source institution.
Each folder contains MR images with the following naming convention:

`TCGA_<institution-code>_<patient-id>_<slice-number>.tif`

Corresponding masks have a `_mask` suffix.

# Notes

整个数据集中一共包含3929张图片，图片尺寸大小为(256, 256)

在目标检测中，如果你想减少结果中的边界框（bbox）数量，主要可以调节以下几个参数：
1. 置信度阈值（Confidence Threshold）：
   - 这是最直接和常用的方法。
   - 提高置信度阈值会过滤掉低置信度的检测结果。
   - 在YOLO中，可以通过 `conf` 参数设置：
     ```python
     results = model(img_path, conf=0.5)  # 设置置信度阈值为0.5
     ```

2. IoU阈值（Intersection over Union Threshold）：
   - 用于非极大值抑制（NMS）过程。
   - 减小IoU阈值会导致更多的重叠框被移除。
   - 在YOLO中，可以通过 `iou` 参数设置：
     ```python
     results = model(img_path, iou=0.5)  # 设置IoU阈值为0.5
     ```

## Logger
yolo默认在训练和测试时会输出大量冗余信息，可以调节其中的logger水平。（通常，第三方库会使用与其包名相同的 logger 名称。）
```python
import logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)
```

如果不确定库使用了哪些 logger，你可以列出所有当前存在的 logger（导入了第三方库，与之相关的logger都包含在其中）：

```python
import logging

# 列出所有 logger
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    print(logger.name)
```

