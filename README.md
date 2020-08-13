# humanpose.pytorch

## Introduction

This is an human pose estimation pytorch implementation derivated from [deep-high-resolution-net.pytorch](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch), aims to achieve lightweight real-time application.

## Features

- [x] It support Distributed DataParallel training, much faster than origin repo.
- [x] support lightweight pose backbones.
- [ ] support lightweight mobile hunman detector.

## Main Results

### Results on MPII val

| Arch               | Head | Shoulder | Elbow | Wrist |  Hip | Knee | Ankle | Mean | Mean@0.1 |
|--------------------|------|----------|-------|-------|------|------|-------|------|----------|
| **pose_hrnet_w32** | 97.067 | 95.686 | 90.21 | 85.644 | 89.077 | 85.795 | 82.711 | 89.927 | 37.931 |
| **pose_hrnet_w48** | 96.930 | 95.771 | 90.864 | 86.329 | 88.731 | 86.862 | 82.829 | 90.208 | 38.002 |

### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch               | Input size | #Params | FLOPs | Weight size | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------------|------------|-------|-------|-------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| **pose_hrnet_w18_v1** | 256x192 | 1.3M  | 0.68G | 5.3M | 0.572 | 0.863 | 0.644 | 0.545 | 0.614 | 0.612 | 0.876 | 0.687 | 0.579 | 0.661 |
| **pose_hrnet_w18_v2** | 256x192 | 3.7M  | 1.8G  | 15M | 0.710 | 0.916 | 0.784 | 0.685 | 0.753 | 0.740 | 0.922 | 0.806 | 0.710 | 0.786 |
| **pose_hrnet_w18_v2_softargmax** | 256x192 | 3.7M  | 1.8G  | 15M | 0.713 | 0.916 | 0.783 | 0.685 | 0.758 | 0.743 | 0.923 | 0.809 | 0.711 | 0.792 |
| **pose_hrnet_w32**    | 256x192 | 28.5M | 7.1G  | 110M | 0.765 | 0.936 | 0.838 | 0.740 | 0.810 | 0.794 | 0.945 | 0.858 | 0.763 | 0.842 |
| **lpn_18**            | 256x192 | 0.47M | 0.42G | 1.9M | 0.445 | 0.773 | 0.445 | 0.434 | 0.467 | 0.497 | 0.798 | 0.519 | 0.474 | 0.531 |
| **lpn_18h**           | 256x192 | 0.50M | 0.43G | 2.1M | 0.486 | 0.806 | 0.506 | 0.472 | 0.511 | 0.533 | 0.821 | 0.567 | 0.508 | 0.570 |
| **lpn_34**            | 256x192 | 0.59M | 0.43G | 2.5M | 0.493 | 0.808 | 0.522 | 0.478 | 0.515 | 0.538 | 0.825 | 0.577 | 0.514 | 0.573 |
| **lpn_50**            | 256x192 | 2.9M | 1.0G | 12M | 0.684 | 0.904 | 0.762 | 0.659 | 0.724 | 0.717 | 0.914 | 0.789 | 0.687 | 0.763 |
| **lpn_100**           | 256x192 | 6.7M | 1.8G | 27M | 0.721 | 0.915 | 0.805 | 0.699 | 0.764 | 0.754 | 0.929 | 0.825 | 0.725 | 0.799 |



## Environment

The code is developed using python 3.6 on Ubuntu 16.04. NVIDIA GPUs are needed. The code is developed and tested using 8 NVIDIA V100 GPU cards. Other platforms or GPU cards are not fully tested.

## Quick start

### Installation

1. Install pytorch >= v1.0.0 following [official instruction](https://pytorch.org/).
2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
3. Install dependencies:

    ```
    pip install -r requirements.txt
    ```
4. Make libs:

   ```
   cd ${POSE_ROOT}/lib
   make
   ```
5. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
6. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── tools 
   ├── README.md
   └── requirements.txt
   ```

7. Download pretrained models from our model zoo([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ))
   ```
   ${POSE_ROOT}
    `-- models
        `-- pytorch
            |-- imagenet
            |   |-- hrnet_w32-36af842e.pth
            |   |-- hrnet_w48-8ef0771d.pth
            |   |-- resnet50-19c8e357.pth
            |-- pose_coco
            |   |-- pose_hrnet_w32_256x192.pth
            |   |-- pose_hrnet_w32_384x288.pth
            |   |-- pose_hrnet_w48_256x192.pth
            |   |-- pose_hrnet_w48_384x288.pth
            |   |-- pose_resnet_50_256x192.pth
            |   `-- pose_resnet_50_384x288.pth
            `-- pose_mpii
                |-- pose_hrnet_w32_256x256.pth
                |-- pose_hrnet_w48_256x256.pth
                `-- pose_resnet_50_256x256.pth

   ```
   
### Data preparation
**For MPII data**, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/). The original annotation files are in matlab format. We have converted them into json format, you also need to download them from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW00SqrairNetmeVu4) or [GoogleDrive](https://drive.google.com/drive/folders/1En_VqmStnsXMdldXA6qpqEyDQulnmS3a?usp=sharing).
Extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- mpii
    `-- |-- annot
        |   |-- gt_valid.mat
        |   |-- test.json
        |   |-- train.json
        |   |-- trainval.json
        |   `-- valid.json
        `-- images
            |-- 000001163.jpg
            |-- 000003072.jpg
```

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. We also provide person detection result of COCO val2017 and test-dev2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing).
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        |   |-- COCO_test-dev2017_detections_AP_H_609_person.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```

### Training and Testing

#### Testing on MPII dataset using model zoo's models([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ))
 

```
python tools/test.py \
    --cfg experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_mpii/pose_hrnet_w32_256x256.pth
```

#### Training on MPII dataset

```
python tools/train.py \
    --cfg experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml
```

#### Testing on COCO val2017 dataset using model zoo's models([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ))
 

```
python tools/test.py \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth \
    TEST.USE_GT_BBOX False
```

#### Training on COCO train2017 dataset

```
python tools/train.py \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
```

### Visualization

#### Visualizing predictions on COCO val

```
python visualization/plot_coco.py \
    --prediction output/coco/w48_384x288_adam_lr1e-3/results/keypoints_val2017_results_0.json \
    --save-path visualization/results

```

## Acknowledgement

* This repo is modified and adapted on these great repositories [deep-high-resolution-net.pytorch](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)

## Contact

```
cavallyb@gmail.com
```

