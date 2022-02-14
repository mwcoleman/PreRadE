
# Activate env
source ~/anaconda3/etc/profile.d/conda.sh
cd ~/Coding/research/mmRad
conda activate lxmert

# Pre-training
python pretrain.py "$@"
