# ICPR Multi-Modal Visual Pattern Recognition Challenge Track 2: Multi-Modal Object Detection

# Overview

<div align=center>
  <img src="pics/pic_0.png" width="300" height="300">
</div>

This is the official repository for Track 2: _Multi-Modal Object Detection challenge (ICPR 2024)_.

This challenge focuses on Object Detection utilizing multi-modal data source including RGB, depth, and infrared images. You can visit the [official website](https://prci-lab.github.io/mmpr-workshop-icpr20242) for more details or directly participate in this track on [codalab]().

# Dataset

In this track, we provide a dataset named **ICPR_JNU MDetection-v1**, which comprises 5,000 multi-modal image pairs (4000 for training and 1000 for testing) across 13 classes. Details are as follows in this repo. To participate in this track, please submit your requirements by choosing "_Challenge Track 2: Multi-modal Detection_" in this [Online Form](https://docs.google.com/forms/d/e/1FAIpQLSeJGZTYW-JS0-IJKnWgYGnE0EgdXnoL7Yi0xc-F9Z6XU1X4Zg/viewform) and filling out orther options.




## Details

  | Lables | Labels Correlation |
  |:-----------:|:-----------:|
  |<img src="pics/labels.jpg" width="300" height="300">| <img src="pics/labels_correlogram.jpg" width="300" height="300"> |

## Examples

| Depth | Thermal-IR | RGB |
|:-----------:|:------------:|:---------:|
| ![Depth Output](pics/depth1.png) | ![IR Output](pics/tir1.png) | ![RGB Output](pics/rgb1.png) |
| ![Depth Output](pics/depth2.png) | ![IR Output](pics/tir2.png) | ![RGB Output](pics/rgb2.png) |

## Structure
```
ICPR_JNU MDetection-v1
├──/images/
│ ├── train
│ │ ├──color
│ │ │ ├── train_0001.png
│ │ │ ├── train_0002.jpg
│ │ │ ├── ... ...
│ │ │ ├── train_4000.jpg
│ │ ├──depth
│ │ │ ├── train_0001.png
│ │ │ ├── train_0002.jpg
│ │ │ ├── ... ...
│ │ │ ├── train_4000.jpg
│ │ ├──infrared
│ │ │ ├── train_0001.png
│ │ │ ├── train_0002.jpg
│ │ │ ├── ... ...
│ │ │ ├── train_4000.jpg
│ ├── val
│ │ ├── ... ...
│ ├── test
│ │ ├──color
│ │ │ ├── test_0001.png
│ │ │ ├── test_0002.jpg
│ │ │ ├── ... ...
│ │ │ ├── test_1000.jpg
│ │ ├──depth
│ │ │ ├── test_0001.png
│ │ │ ├── test_0002.jpg
│ │ │ ├── ... ...
│ │ │ ├── test_1000.jpg
│ │ ├──infrared
│ │ │ ├── test_0001.png
│ │ │ ├── test_0002.jpg
│ │ │ ├── ... ...
│ │ │ ├── test_1000.jpg
└──/labels/
  ├── /train/color/
  │ ├── train_0001.txt
  │ ├── train_0002.txt
  │ │ ├── ... ...
  │ ├── train_4000.txt
  ├── /val/color
  │ ├── ... ...
  ├── /test/color/
  │ ├── test_0001.txt
  │ ├── test_0002.txt
  │ │ ├── ... ...
  │ ├── test_1000.txt
  └───
```

# Baseline

This code is based on [yolo-v5](https://github.com/ultralytics/yolov5/releases), you can follow the [Readme_yolo]() first to build a suitable envirement.  We have modified it to accommodate this multimodal task, while you can also build your own model to accomplish this task.


## Training 






