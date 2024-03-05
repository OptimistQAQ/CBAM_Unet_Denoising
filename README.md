# CBAM_Unet_Denoising
Few-shot RAW Image Denoising @ MIPI-challenge

## Title: DeUnet: An Unet++ Network for Low-light Denoising

### Prerequisites

+ Python >== 3.6, PyTorch >= 1.6
+ Requirements: opencv-python, rawpy, exifread, h5py, scipy
+ Platforms: Ubuntu 16.04, cuda-10.1
+ Our method can run on the CPU, but we recommend you run it on the GPU

## Quick Start

1. Train

```shell
python3 trainer_My.py -f runfiles/My.yml --mode train
```

2. test

```shell
python3 test.py
```

## Model Download

Our model: 

[Model](https://drive.google.com/drive/folders/16LPan408Pp0jAWAhLiuw8_wgQYUj4MMp?usp=drive_link)

## Test Data

Test Data:

[Data](https://drive.google.com/drive/folders/16LPan408Pp0jAWAhLiuw8_wgQYUj4MMp?usp=drive_link)

