#!/bin/bash
#SBATCH -N 1
#SBATCH --time=48:00:00
#SBATCH --mem=16000
#SBATCH -o logfiles/test-output_dtermen_rerun.out
#SBATCH -e logfiles/test-error_dtermen_rerun.out

#SBATCH --array=0-63

FILES=(/home/gridsan/groups/keatinglab/mlu/bcl2/clean_pdb/*)
FILE=${FILES[$SLURM_ARRAY_TASK_ID]}
NAME=${FILE##*/}
BASE=${NAME%.pdb}

. /etc/profile.d/modules.sh
mkdir ${BASE}
cd ${BASE}
/home/gridsan/groups/keatinglab/MST_workspace/MST/bin/design --p ${FILE} --c /home/gridsan/groups/keatinglab/dTERMen/config_files/design_clique1.configfile --o ${BASE} --w > design.o
cd ..
