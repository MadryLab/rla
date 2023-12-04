#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=xeon-p8
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o OUTPUTDIR/etab-output.out
#SBATCH -e OUTPUTDIR/etab-error.out

# activate conda
CONDA_ROOT=/state/partition1/llgrid/pkg/anaconda/anaconda3-2019b/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate terminator

python to_etab.py \
    --output_dir=OUTPUTDIR \
    --dtermen_data=DTERMENDATA \
    --num_cores=64 -u
python batch_arr_dTERMen.py \
    --output_dir=OUTPUTDIR \
    --pdb_root=PDBROOT \
    --dtermen_data=DTERMENDATA \
    --batch_size=48
