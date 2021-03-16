#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-03:00
#SBATCH --output=train_base-%N-%j.out

module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch --no-index

python pytorch-test.py

python train.py \
	--eval_every 1 \
	--vis_every 1 \
	--conf ../experiments/nuScenes/models/vel_ee/config.json \
	--train_data_dict nuScenes_train_full.pkl \
	--eval_data_dict nuScenes_val_full.pkl \
	--offline_scene_graph yes \
	--preprocess_workers 10 \
	--batch_size 256 \
	--log_dir ../experiments/nuScenes/models \
	--train_epochs 20 \
	--node_freq_mult_train \
	--log_tag _vel_ee \
	--augment \
