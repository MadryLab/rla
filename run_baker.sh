#!/bin/bash
#SBATCH --mincpu=1
#SBATCH --gres=gpu:volta:1
#SBATCH --time=96:00:00
#SBATCH -o /home/gridsan/fbirnbaum/joint-protein-embs/train-output_run_VirB8.out
#SBATCH -e /home/gridsan/fbirnbaum/joint-protein-embs/train-error_run_VirB8.out

CONDA_ROOT=/state/partition1/llgrid/pkg/anaconda/anaconda3-2022b/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate terminator_fine_tune
export LOCAL_RANK=-1
ulimit -s unlimited
ulimit -n 10000

python /home/gridsan/fbirnbaum/joint-protein-embs/contact_baker.py --target VirB8


