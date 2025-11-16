# SLab Medical Segmentation (SMS)

A general medical segmentation toolkit developed by the SLab team, led by Prof. Shuang Song.

## :rocket: News
* **(2025.10.16):** The first version of SLab Medical Segmentation (SMS) repo are released.


## Table of Contents
Follow steps 1-3 to run our toolkit. 

1. [Introduction](#Introduction)
1. [Installation](#Installation) 
2. [Prepare the Dataset](#Prepare-the-Dataset)
3. [Run the Experiment](#Run-the-Experiment)
4. [Visualization](#Visualization)
5. [Results](#Results)
6. [To Do List](#TODO)
7. [Acknowledgements](#Acknowledgements)

## Introduction

Support networks:


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

----


* [TransBTS](https://arxiv.org/abs/2103.04430) (MICCAI2021, [code](https://github.com/Rubics-Xuan/TransBTS))



Support dataset:
* 

## Installation

conda create -n slab_med_seg python=3.10
conda activate slab_med_seg
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

## Prepare the Dataset


## Run the Experiment
```
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
## Results 
We benchmark SLab Medical Segmentation (SMS) both qualitatively and quantitatively against the current state-of-the-art models such as xx and several other models on three widely used medical datasets: BRaTs2017, xxx


### Brain Tumor Segmentation Dataset (BraTs)



## TODO
* pretrained finetuning


## Acknowledgements
* https://github.com/yhygao/CBIM-Medical-Image-Segmentation