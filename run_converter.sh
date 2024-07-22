#!/bin/bash
#SBATCH --mincpu=20
#SBATCH --gres=gpu:volta:1
#SBATCH --time=96:00:00
#SBATCH -o /home/gridsan/fbirnbaum/joint-protein-embs/converter_runs/multichain_lr_5_reg_3/log.out
#SBATCH -e /home/gridsan/fbirnbaum/joint-protein-embs/converter_runs/multichain_lr_5_reg_3/log.err

CONDA_ROOT=/state/partition1/llgrid/pkg/anaconda/anaconda3-2022b/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate terminator_fine_tune
ulimit -s unlimited
ulimit -n 10000

python /home/gridsan/fbirnbaum/joint-protein-embs/train_converter.py --run_dir /home/gridsan/fbirnbaum/joint-protein-embs/converter_runs/multichain_lr_5_reg_3 --train_wds multichain_clip_train.wds --val_wds multichain_clip_val.wds --lr 1e-5 --regularization 0.001 --dev cuda:0