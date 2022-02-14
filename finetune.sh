#!/bin/bash

encoder=$1
name=$2
#PROGRAM LEVEL
#encoder=uclanlp/visualbert-vqa-coco-pre
#name=baseline
#if [ -z "$1" ]
#then
#echo IN-LOOP
#encoder=$1
#name=$(echo $encoder | cut -d'/' -f 7)
#fi
#checkpoint=None
#log_offline=True

#TRAINING
epochs=10

#MODEL
max_seq_len=125
batch_size=64

#DATA
dataset=mimic
topk=0
data_path=/media/matt/data21/mmRad/MIMIC
img_path=mimic_train_100k.tsv
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
    --max_seq_len $max_seq_len \
    --batch_size $batch_size \
    --dataset $dataset \
    --topk $topk \
    --data_path $data_path \
    --img_path $img_path \
    --txt_path $txt_path \
    --test_data $test_data \
    #--log_offline $log_offline \
    #--load_cp_path $load_cp
