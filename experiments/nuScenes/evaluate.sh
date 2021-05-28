#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --job-name=evaluate
#SBATCH --output=/home/cchen795/scratch/slurm/%x-%j.out
#SBATCH --error=/home/cchen795/scratch/slurm/%x-%j.out
#SBATCH --mem-per-cpu=32G

SCRATCH=/scratch/cchen795
echo "load modules"
module load python/3.6
module load ipython-kernel/3.6
module load geos

echo "activate virtualenv"
source $HOME/pytrajectron/bin/activate
APPDIR=$SCRATCH/code/trajectron-plus-plus/experiments/nuScenes

PH=2
MODEL_NAME="models_19_Mar_2021_22_14_19_int_ee_me_ph8"
TAG_NAME="models_19_Mar_2021_22_14_19_int_ee_me_ph${PH}_ckpt20"
echo "start model evaluate"
python $APPDIR/evaluate.py \
	--model $APPDIR/models/$MODEL_NAME \
	--data $SCRATCH/nuScenes/processed/nuScenes_test_full.pkl \
	--checkpoint=20 \
	--output_path $APPDIR/results \
	--output_tag $TAG_NAME \
	--node_type VEHICLE \
	--prediction_horizon $PH

