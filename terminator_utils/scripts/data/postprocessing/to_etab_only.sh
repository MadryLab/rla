#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=xeon-p8
#SBATCH --time=02:00:00
#SBATCH --mem=0
#SBATCH -o OUTPUTDIR/etab-output.out
#SBATCH -e OUTPUTDIR/etab-error.out

# activate conda
CONDA_ROOT=/state/partition1/llgrid/pkg/anaconda/anaconda3-2022b/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate ffcv_cfs

python to_etab.py \
    --output_dir=OUTPUTDIR \
    --num_cores=64 -u
