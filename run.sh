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
                --loss_type combine \
                