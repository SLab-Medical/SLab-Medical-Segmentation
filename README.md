# SLab Medical Segmentation (SMS)

A advanced and general medical segmentation toolkit developed by the SLab team, led by Prof. Shuang Song and Kangneng Zhou.

## :rocket: News
* **(2025.11.20):** What a beautiful day! The first version of the SLab Medical Segmentation (SMS) repository has been released.


## Table of Contents
Follow steps 1-3 to run our toolkit.

1. [Introduction](#Introduction)
2. [Installation](#Installation)
3. [Prepare the Dataset](#Prepare-the-Dataset)
4. [Run the Training](#Run-the-Training)
5. [Run the Inference](#Run-the-Inference)
6. [Visualization](#Visualization)
7. [Results](#Results)
8. [To Do List](#TODO)
9. [By The Way](#By-The-Way)
10. [Acknowledgements](#Acknowledgements)

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
pip install torch torchvision
pip install -r requirements.txt
```

## Prepare the Dataset
design your dataloader in dataset

## Run the Training

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

## Run the Inference
```bash
python inference.py \
    --checkpoint_path logs/exp/best_model.pth \
    --input_path data/test/img/ \
    --output_dir results/ \
    --config logs/exp/config.yaml \
    --use_sliding_window \
    --calculate_metrics \
    --save_probability
```

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

