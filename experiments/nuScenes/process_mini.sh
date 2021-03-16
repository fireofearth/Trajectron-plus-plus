#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --job-name=process_data
#SBATCH --output=~/scratch/slurm/process_data-%x-%j.out
#SBATCH --error=~/scratch/slurm/process_data-%x-%j.out

## Tutorial
# --mem-per-cpu (memory per core) defaults to 256 MB

echo "activate virtualenv"
source ~/pytraj
APPDIR=/home/cchen795/code/trajectron-plus-plus/experiments/nuScenes
python --version
echo "start process_data"
python $APPDIR/process_data.py \
	--data=~/scratch/nuScenes/v1.0-mini \
	--version="v1.0-mini" \
	--output_path=~/scratch/nuScenes/processsed
