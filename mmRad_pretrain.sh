#!/bin/bash

# The name of experiment
name=PT-mlmitm-12hr-benchmark
encoder=None #uclanlp/visualbert-vqa-coco-pre
dataset=mimic_train
tasks=\[\'mlm\'\,\'itm\'\]
epochs=180
load_cp=/media/matt/data21/mmRad/checkpoints/PT/PT-mlmitm-12hr-benchmark/pl_framework/epoch=23-step=9191.ckpt
# Activate env
source ~/anaconda3/etc/profile.d/conda.sh
cd ~/Coding/research/mmRad
conda activate lxmert

# Pre-training
python pretrain.py \
    --name $name \
    --train $dataset \
    --epochs $epochs --topk 0 \
    --num_attention_heads 12 --num_tx_layers 12 \
    --batch_size 128 --lr 5e-5 \
    --max_seq_len 50 \
    --tasks $tasks --load_cp_path $load_cp

