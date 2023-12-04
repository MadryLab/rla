#!/bin/bash
#SBATCH -N 1
#SBATCH --time=72:00:00
#SBATCH --mem=16000
#SBATCH -o logfiles/test-output_dtermen-%x-%j.out
#SBATCH -e logfiles/test-error_dtermen-%x-%j.out

. /etc/profile.d/modules.sh
FILE=$1
NAME=${FILE##*/}
BASE=${NAME%.pdb}

mkdir ${BASE}
cd ${BASE}
/home/gridsan/groups/keatinglab/MST_workspace/MST/bin/design --p ${FILE} --c /home/gridsan/groups/keatinglab/dTERMen/config_files/design_default.configfile --s "chain B" --o ${BASE} --w > design.o
cd ..
