#!/bin/bash

. /etc/profile.d/modules.sh

list_file=/data1/groups/keating_madry/baseline_val_output/sequence_design/baseline_val_output_etabs_list.txt # a file of elements (values, paths, etc) each \n separated
readarray -t RUNLIST < "$list_file"
RUNLIST_LEN=${#RUNLIST[@]}
batch_size=$(($(($RUNLIST_LEN/$LLSUB_SIZE))+1))
runfile=/home/gridsan/sswanson/local_code_mirror/joint-protein-embs/terminator_utils/scripts/data/postprocessing/design_complex_penalt.sh # the script that will be run for each element in the file

# get the batch boundaries
let start=$batch_size*$LLSUB_RANK
let next=$LLSUB_RANK+1
let next=$batch_size*$next
if [[ $next -gt $RUNLIST_LEN ]]
then
        let end=$RUNLIST_LEN
else
        let end=$next
fi

echo "list length: "$RUNLIST_LEN
echo "total workers: "$LLSUB_SIZE
echo "batch size: "$batch_size
echo "worker ID: "$LLSUB_RANK
echo "batch start: "$start
echo "batch end: "$end

SECONDS=0
# run the batch
i=$start
while [[ $i -lt $end ]]
do
        element=${RUNLIST[$i]}
        echo bash $runfile $element
        bash $runfile $element
        i=$(($i + 1))
done

ELAPSED="Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo $ELAPSED