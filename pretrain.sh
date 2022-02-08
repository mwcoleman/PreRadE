#!/bin/bash

tasks=\[\'mlm\'\,\'itm\'\]

#PROGRAM LEVEL
name=$tasks
encoder=uclanlp/visualbert-vqa-coco-pre
#checkpoint=None
save_cp_path=/media/matt/data21/mmRad/checkpoints/

#TRAINING
epochs=200
max_hrs=12


#MODEL
max_seq_len=125
batch_size=64

#DATA
dataset=mimic
topk=512
data_path=/media/matt/data21/mmRad/MIMIC
img_path=mimic_train_100k.tsv
txt_path=studies_with_splits.csv

# Activate env
source ~/anaconda3/etc/profile.d/conda.sh
cd ~/Coding/research/mmRad
conda activate lxmert

# Pre-training
python pretrain.py \
    --name $name \
    --load_model $encoder \
    --epochs $epochs \
    --max_hrs $max_hrs \
    --tasks $tasks \
    --max_seq_len $max_seq_len \
    --batch_size $batch_size \
    --dataset $dataset \
    --topk $topk \
    --data_path $data_path \
    --img_path $img_path \
    --txt_path $txt_path \
    --save_cp_path $save_cp_path
    #--log_offline $log_offline \
    #--load_cp_path $load_cp
