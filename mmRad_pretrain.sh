#!/bin/bash

# The name of experiment
name=PT-MIMIC-mlmmfr-full
encoder=None
dataset=mimic_train
tasks=\[\'mlm\'\,\'mfr\'\]
# Activate env
source ~/anaconda3/etc/profile.d/conda.sh
cd ~/Coding/research/mmRad
conda activate lxmert

# Pre-training
python pretrain.py \
    --name $name \
    --train $dataset \
    --epochs 30 --topk None \
    --load_model $encoder \
    --num_attention_heads 12 --num_tx_layers 12 \
    --batch_size 256 --lr 2e-4

