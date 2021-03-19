#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=evaluate_dynmap
#SBATCH --output=/scratch/cchen795/slurm/%x-%j.out
#SBATCH --error=/scratch/cchen795/slurm/%x-%j.out

SCRATCH=/scratch/cchen795
echo "load modules"
module load python/3.6
module load ipython-kernel/3.6
module load geos

echo "activate virtualenv"
source $HOME/pytrajectron/bin/activate
APPDIR=$SCRATCH/code/trajectron-plus-plus/experiments/nuScenes

echo "evaluate"
python $APPDIR/evaluate.py \
    --model models/int_ee_me \
    --checkpoint=12 \
    --data $SCRATCH/nuScenes/processed/nuScenes_test_full.pkl \
    --output_path results \
    --output_tag int_ee_me \
    --node_type VEHICLE \
    --prediction_horizon 6
