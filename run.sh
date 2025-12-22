conda activate seg

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



python train.py --model_name 3DUXNET 




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


python train.py \
    --dataset_name dsa \
    --dataset_path data/data_2d/DSA \
    --model_name unet_2d \
    --dimension 2d \
    --in_channels 1 \
    --out_channels 1 \
    --batch 4 \
    --num_epochs 100 \
    --expname dsa_unet_2d