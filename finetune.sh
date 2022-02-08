#!/bin/bash

# The name of experiment
name=FT-coco-uclan
encoder=$1
dataset=mimic_train
# Activate env
source ~/anaconda3/etc/profile.d/conda.sh
cd ~/Coding/research/mmRad
conda activate lxmert

# Pre-training
python finetune.py \
    --name $name \
    --train $dataset \
    --epochs 30 --topk None \
    --load_model $encoder \
    --num_attention_heads 12 --num_tx_layers 12 \
    --batch_size 256 --lr 5e-5 --topk 20480

