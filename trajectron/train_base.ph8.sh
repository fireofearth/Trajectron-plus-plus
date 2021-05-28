#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=train_base_ph8
#SBATCH --output=/scratch/cchen795/slurm/%x-%j.out
#SBATCH --error=/scratch/cchen795/slurm/%x-%j.out

SCRATCH=/scratch/cchen795
echo "load modules"
module load python/3.6
module load ipython-kernel/3.6
module load geos

echo "activate virtualenv"
source $HOME/pytrajectron/bin/activate
APPDIR=$SCRATCH/code/trajectron-plus-plus/trajectron

echo "train"
export MPLBACKEND="agg"
python train.py \
	--data_dir $SCRATCH/nuScenes/processed \
	--eval_every 1 \
	--vis_every 1 \
	--conf ./vel_ee.ph8.config.json \
	--train_data_dict nuScenes_train_full.pkl \
	--eval_data_dict nuScenes_val_full.pkl \
	--offline_scene_graph yes \
	--preprocess_workers 10 \
	--batch_size 256 \
	--log_dir ../experiments/nuScenes/models \
	--train_epochs 20 \
	--node_freq_mult_train \
	--log_tag _vel_ee_ph8 \
	--augment \
