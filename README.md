# SLab Medical Segmentation (SMS)

A advanced and general medical segmentation toolkit developed by the SLab team, led by Prof. Shuang Song and Kangneng Zhou.

## :rocket: News
* **(2025.11.20):** What a beautiful day! The first version of the SLab Medical Segmentation (SMS) repository has been released.


## Table of Contents
Follow steps 1-3 to run our toolkit.

1. [Introduction](#Introduction)
1. [Installation](#Installation)
2. [Prepare the Dataset](#Prepare-the-Dataset)
3. [Run the Experiment](#Run-the-Experiment)
4. [Visualization](#Visualization)
5. [Results](#Results)
6. [To Do List](#TODO)
7. [By The Way](#By-The-Way)
8. [Acknowledgements](#Acknowledgements)

## Introduction

### Support 3D Networks:

* Unetr
* unet
* unetr_pp
* [Segformer3D](https://arxiv.org/abs/2404.10156) (CVPR-W 2024, [code](https://github.com/OSUPCVLab/SegFormer3D))
* [SegMamba](https://arxiv.org/pdf/2401.13560) (ICCV2025, [code](https://github.com/ge-xing/SegMamba))
* [Slim-UNETR](https://ieeexplore.ieee.org/document/10288609) (TMI 2023, [code](https://github.com/deepang-ai/Slim-UNETR))
* vtunet
* attention_unet
* SwinUNETR
* unet++
* [3DUXNET](https://arxiv.org/abs/2209.15076) (ICLR2023, [code](https://github.com/MASILab/3DUX-Net))
* nnFormer
* [EfficientMedNext](https://papers.miccai.org/miccai-2025/0280-Paper4895.html) (MICCAI2025, [code](https://github.com/SLDGroup/EfficientMedNeXt))
* nnUnet
* vnet
* medformer
* [TransBTS](https://arxiv.org/abs/2103.04430) (MICCAI2021, [code](https://github.com/Rubics-Xuan/TransBTS))

### Support 2D Networks:

* unet - Standard 2D U-Net
* unetpp - U-Net++ (ResNet34 backbone)
* deeplab - DeepLabV3 (ResNet101 + ASPP)
* fcn - FCN32s
* segnet - SegNet
* pspnet - PSPNet
* highresnet - HighResNet
* miniseg - MiniSeg






### Support Datasets:
* Brain Tumor Segmentation Dataset (BraTS) - 3D
* DSA (Digital Subtraction Angiography) - 2D

## Installation

```
conda create -n slab_med_seg python=3.10
conda activate slab_med_seg
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Prepare the Dataset


## Run the Experiment

### For 3D Models:
```bash
python train.py --dataset_name your_dataset_name \
                --dataset_path your_dataset_path \
                --batch 4 \
                --num_workers 4 \
                --queue_length 5 \
                --samples_per_volume 5 \
                --patch_size 96 96 96 \
                --expname your_exp_name \
                --model_name your_model_name \
                --in_channels 1 \
                --out_channels 3 \
                --loss_type combine
```

### For 2D Models:
```bash
python train.py --dataset_name dsa \
                --dataset_path data/data_2d/DSA \
                --model_name unet \
                --dimension 2d \
                --in_channels 1 \
                --out_channels 1 \
                --batch 4 \
                --num_epochs 100 \
                --expname dsa_unet_2d \
                --loss_type combine
```

**Note:**
- 2D models use standard PyTorch DataLoader with bilinear interpolation
- 3D models use TorchIO SubjectsLoader with trilinear interpolation
- See `README_2D.md` for detailed 2D usage instructions
## Results 
We benchmark SLab Medical Segmentation (SMS) both qualitatively and quantitatively against the current state-of-the-art models such as MedNeXt and several other models on three widely used medical datasets: BRaTs2017.


### Brain Tumor Segmentation Dataset (BraTs)
Stay tuned!


## TODO
* More datasets

## By The Way
If you're using this project and have feedback, feel free to contact Kangneng Zhou via WeChat (kangkangellis666) or email (elliszkn@163.com).

## Acknowledgements
This repository offers advanced and general medical segmentation in 3D and 2D. The project was completed under the supervision of Prof. Shuang Song and Kangneng Zhou. Special thanks to Haoyu Yuan, Dian Song, Jingdan Zhang, Yihang Xu and Sibo Zhao for their invaluable support.

