#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=xeon-p8
#SBATCH --time=0:20:00
#SBATCH -o /dev/null

. /etc/profile.d/modules.sh
ids_file=$2
batch_size=$3
readarray -t RUNLIST < $ids_file
RUNLIST_LEN=${#RUNLIST[@]}

let start=$batch_size*$SLURM_ARRAY_TASK_ID
# compute the end of the batch
let next=$SLURM_ARRAY_TASK_ID+1
let next=$batch_size*$next
if [[ $next -gt $RUNLIST_LEN ]]
then
  let end=$RUNLIST_LEN
else
  let end=$next
fi

output_dir=$1
cd $1

# run the batch
i=$start
while [[ $i -lt $end ]]
do
  ID=${RUNLIST[$i]}
  echo $ID
  runfile="run_${ID}.sh"
  bash $runfile > "${ID}-output.out" 2> "${ID}-error.err" &
  i=$(($i + 1))
done

wait