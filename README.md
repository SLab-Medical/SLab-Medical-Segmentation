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

* [Unetr](https://arxiv.org/abs/2103.10504) (WACV 2022, [code](https://monai.io/research/unetr))
* [unet](https://arxiv.org/abs/1606.06650) (MICCAI 2016, [code](https://github.com/wolny/pytorch-3dunet))
* [unetr_pp](https://arxiv.org/abs/2212.04497) (TMI 2024, [code](https://github.com/Amshaker/unetr_plus_plus))
* [Segformer3D](https://arxiv.org/abs/2404.10156) (CVPR-W 2024, [code](https://github.com/OSUPCVLab/SegFormer3D))
* [SegMamba](https://arxiv.org/pdf/2401.13560) (ICCV 2025, [code](https://github.com/ge-xing/SegMamba))
* [Slim-UNETR](https://ieeexplore.ieee.org/document/10288609) (TMI 2023, [code](https://github.com/deepang-ai/Slim-UNETR))
* [vtunet](https://arxiv.org/abs/2111.13300) (MICCAI 2022, [code](https://github.com/himashi92/VT-UNet))
* [AttentionUnet](https://arxiv.org/abs/1804.03999) (MIDL 2018, [code](https://github.com/sfczekalski/attention_unet))

* [SwinUNETR](https://arxiv.org/abs/2201.01266) (MICCAI-W 2021, [code](https://github.com/LeonidAlekseev/Swin-UNETR)) 
* [UNet++](https://arxiv.org/abs/1807.10165) (TMI 2019, [code](https://github.com/MrGiovanni/UNetPlusPlus))
* [3DUXNET](https://arxiv.org/abs/2209.15076) (ICLR 2023, [code](https://github.com/MASILab/3DUX-Net))
* [nnFormer](https://ieeexplore.ieee.org/document/10183842) (TIP 2023, [code](https://github.com/282857341/nnFormer))
* [EfficientMedNext](https://papers.miccai.org/miccai-2025/0280-Paper4895.html) (MICCAI 2025, [code](https://github.com/SLDGroup/EfficientMedNeXt))
* [nnUnet](https://www.nature.com/articles/s41592-020-01008-z) (Nature Methods 2021, [code](https://github.com/MIC-DKFZ/nnUNet))
* [VNet](https://arxiv.org/abs/1606.04797) (3DV 2016, [code](https://github.com/faustomilletari/VNet))
* [MedFormer](https://arxiv.org/abs/2203.001310) (arXiv 2022, [code](https://github.com/yhygao/CBIM-Medical-Image-Segmentation))
* [TransBTS](https://arxiv.org/abs/2103.04430) (MICCAI2021, [code](https://github.com/Rubics-Xuan/TransBTS))

### Support 2D Networks:

* [UNet](https://arxiv.org/abs/1505.04597) (MICCAI 2015, [code](https://github.com/milesial/Pytorch-UNet))
* [UNet++](https://arxiv.org/abs/1807.10165) (TMI 2019, [code](https://github.com/MrGiovanni/UNetPlusPlus))
* [DeepLabV3](https://arxiv.org/abs/1706.05587) (arXiv 2017, [code](https://github.com/VainF/DeepLabV3Plus-Pytorch))
* [FCN](https://arxiv.org/abs/1411.4038) (CVPR 2015, [code](https://github.com/shelhamer/fcn.berkeleyvision.org))
* [SegNet](https://arxiv.org/abs/1511.00561) (TPAMI 2017, [code](https://github.com/delta-onera/segnet_pytorch))
* [PSPNet](https://arxiv.org/abs/1612.01105) (CVPR 2017)
* [HighResNet](https://arxiv.org/abs/1707.01992) (IPMI 2017, [code](https://github.com/fepegar/highresnet))
* [MiniSeg](https://arxiv.org/abs/2004.09750) (TNNLS 2022, [code](https://github.com/yun-liu/MiniSeg))
* [AttentionUnet](https://arxiv.org/abs/1804.03999) (MIDL 2018, [code](https://github.com/yhygao/CBIM-Medical-Image-Segmentation))
* [DANet](https://arxiv.org/abs/1809.02983) (CVPR 2019, [code](https://github.com/junfu1115/DANet))
* [MedFormer](https://arxiv.org/abs/2203.001310) (arXiv 2022, [code](https://github.com/yhygao/CBIM-Medical-Image-Segmentation))
* [TransUnet](https://arxiv.org/abs/2102.04306) (MIA 2024, [code](https://github.com/Beckschen/TransUNet))
* [SwinUNet](https://arxiv.org/abs/2105.05537) (ECCV-W 2024, [code](https://github.com/HuCaoFighting/Swin-Unet))




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

