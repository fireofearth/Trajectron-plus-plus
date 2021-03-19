#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --job-name=process_data
#SBATCH --output=/home/cchen795/scratch/slurm/%x-%j.out
#SBATCH --error=/home/cchen795/scratch/slurm/%x-%j.out
#SBATCH --mem-per-cpu=32G

## Tutorial
# --mem-per-cpu (memory per core) defaults to 256 MB
# can't use ~ tilde in SBATCH header to refer to home directory
# the cwd of the script is the same as where the user called sbatch
# 

SCRATCH=/scratch/cchen795
echo "load modules"
module load python/3.6
module load ipython-kernel/3.6
module load geos

echo "activate virtualenv"
source $HOME/pytrajectron/bin/activate
APPDIR=$SCRATCH/code/trajectron-plus-plus/experiments/nuScenes

#echo "debug"
#pwd
#python --version
#pip freeze

echo "start process_data"
python $APPDIR/process_data.py \
	--data=$SCRATCH/nuScenes/v1.0 \
	--version="v1.0-trainval" \
	--output_path=$SCRATCH/nuScenes/processed
