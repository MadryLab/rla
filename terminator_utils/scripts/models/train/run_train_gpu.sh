#!/bin/bash
#SBATCH -N 1
#SBATCH --mincpu=20
#SBATCH --gres=gpu:volta:1
#SBATCH --time=1-00:00:00
#SBATCH --mem=50G
#SBATCH -o RUNDIR/train.%j.out
#SBATCH -e RUNDIR/train.%j.err

CONDA_ROOT=/state/partition1/llgrid/pkg/anaconda/anaconda3-2022b/
conda activate ffcv_cfs
echo $CONDA_PREFIX
ulimit -s unlimited
ulimit -n 10000
module load cuda/11.1
module load nccl/2.8.3-cuda11.1
export NCCL_DEBUG=INFO

TERMINATOR_REPO=$1
echo $2
echo $3
echo $4
echo $5

python $TERMINATOR_REPO/terminator_utils/scripts/models/train/train.py \
  --dataset=$2/multichain_clean \
  --model_hparams=$3 \
  --run_hparams=$4 \
  --run_dir=$5 \
  --train=$2/train.in \
  --validation=$2/validation.in \
  --test=$2/test.in \
  --lazy
