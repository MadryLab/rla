#!/bin/bash
#SBATCH --mincpu=47
#SBATCH --time=96:00:00
#SBATCH -o /home/gridsan/fbirnbaum/joint-protein-embs/baker_finetune_runs/IL7Ra-FGFR2-InsulinR-PDGFR_lr_5_reg_3_nostop_200_tr_5_val_5_small_drop_2/log.out
#SBATCH -e /home/gridsan/fbirnbaum/joint-protein-embs/baker_finetune_runs/IL7Ra-FGFR2-InsulinR-PDGFR_lr_5_reg_3_nostop_200_tr_5_val_5_small_drop_2/log.err

CONDA_ROOT=/state/partition1/llgrid/pkg/anaconda/anaconda3-2022b/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate terminator_fine_tune
ulimit -s unlimited
ulimit -n 10000

python /home/gridsan/fbirnbaum/joint-protein-embs/binder_finetune.py --run_dir /home/gridsan/fbirnbaum/joint-protein-embs/baker_finetune_runs/IL7Ra-FGFR2-InsulinR-PDGFR_lr_5_reg_3_nostop_200_tr_5_val_5_small_drop_2 --train_targets InsulinR,FGFR2,IL7Ra,PDGFR --val_targets InsulinR,FGFR2,IL7Ra,PDGFR --lr 1e-5 --regularization 0.001 --epochs 100 --train_cut 0.5 --val_cut 0.5 --split_target True --in_features 1000 --out_features 1 --num_layers 1 --dropout 0.2 --dev cpu


###SBATCH --gres=gpu:volta:1


