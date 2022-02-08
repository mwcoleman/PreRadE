#!/bin/bash

#PROGRAM LEVEL
name=FT-baseline
encoder=uclanlp/visualbert-vqa-coco-pre
#checkpoint=None
log_offline=True

#TRAINING
epochs=10
max_hrs=12

#MODEL
max_seq_len=125
batch_size=64

#DATA
dataset=mimic
topk=5120
data_path=/media/matt/data21/mmRad/MIMIC
img_path=mimic_val_100k.tsv
txt_path=studies_with_splits.csv
test_data=mimic_test_100k.tsv

# Activate env
source ~/anaconda3/etc/profile.d/conda.sh
cd ~/Coding/research/mmRad
conda activate lxmert

# Pre-training
python finetune.py \
    --name $name \
    --load_model $encoder \
    --epochs $epochs \
    --max_hrs $max_hrs \
    --max_seq_len $max_seq_len \
    --batch_size $batch_size \
    --dataset $dataset \
    --topk $topk \
    --data_path $data_path \
    --img_path $img_path \
    --txt_path $txt_path \
    --test_data $test_data
    #--log_offline $log_offline \
    #--load_cp_path $load_cp
