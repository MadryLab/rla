#!/bin/bash
#SBATCH -N 1
#SBATCH --mincpu=16
#SBATCH --gres=gpu:volta:2
#SBATCH --time=HOURS:00:00
#SBATCH --mem=35G
#SBATCH --exclusive
#SBATCH -o RUNDIR/train-output_runRUNNO.out
#SBATCH -e RUNDIR/train-error_runRUNNO.out

CONDA_ROOT=/state/partition1/llgrid/pkg/anaconda/anaconda3-2022b/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate terminator
ulimit -s unlimited
ulimit -n 10000

python train.py \
  --dataset=DATASET \
  --model_hparams=MODEL_HPARAMS \
  --run_hparams=RUN_HPARAMS \
  --run_dir=RUNDIR \
  --out_dir=OUTPUTDIR \
  --dev=cpu \
  --epochs=100 \
  --train=DATASET/TRAIN.in \
  --validation=DATASET/VALIDATION.in \
  --test=DATASET/TEST.in
