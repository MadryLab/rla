#!/bin/bash
. /etc/profile.d/modules.sh

# Set variables
TERMINATOR_REPO=/home/gridsan/sswanson/local_code_mirror/joint-protein-embs
DATASET=/data1/groups/keating_madry/multichain_data
MODEL_HPARAMS=$TERMINATOR_REPO/terminator_configs/coordinator.json
RUN_HPARAMS=$TERMINATOR_REPO/terminator_configs/coordinator_run_finetune.json
RUNDIR=$PWD

sbatch $TERMINATOR_REPO/terminator_utils/scripts/models/train/run_train_gpu.sh $TERMINATOR_REPO $DATASET $MODEL_HPARAMS $RUN_HPARAMS $RUNDIR