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

* [Segformer3D](https://arxiv.org/abs/2404.10156) (CVPR-W 2024, [code](https://github.com/OSUPCVLab/SegFormer3D)) 
* [Slim-UNETR](https://ieeexplore.ieee.org/document/10288609) (TMI 2023, [code](https://github.com/deepang-ai/Slim-UNETR)) 
* [SegMamba](https://arxiv.org/pdf/2401.13560) (ICCV2025, [code](https://github.com/ge-xing/SegMamba))
* EfficientMedNext
* 3DUXNET


Support dataset:
* 
* 



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
