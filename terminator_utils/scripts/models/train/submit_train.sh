#!/bin/bash

# collect args
DATASET=$(readlink -f $1)
DATANAME=${1##*/}
MODEL_HPARAMS=$(readlink -f $2)
RUN_HPARAMS=$(readlink -f $3)
RUNDIR=$(readlink -f $4)
RUNNAME=${4##*/}
OUTPUTDIR=$(readlink -f $5)
HOURS=$6
TRAIN=$7
VALIDATION=$8
TEST=$9
echo "$DATANAME $RUNNAME $OUTPUTDIR"

# compute what directory this file is in
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
# DIR is the directory this file is in, e.g. postprocessing
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

cd $DIR

# create the folder to store the submission script
if [[ ! -d bash_files ]];
then
  mkdir bash_files
fi

# create the run dir and output dir
if [[ ! -d $RUNDIR ]];
then
  mkdir $RUNDIR
fi
if [[ ! -d $OUTPUTDIR ]];
then
  mkdir $OUTPUTDIR
fi

sed \
  -e "s|DATASET|${DATASET}|g" \
  -e "s|DATANAME|${DATANAME}|g" \
  -e "s|RUNNO|${RUNNO}|g" \
  -e "s|MODEL_HPARAMS|${MODEL_HPARAMS}|g" \
  -e "s|RUN_HPARAMS|${RUN_HPARAMS}|g" \
  -e "s|RUNDIR|${RUNDIR}|g" \
  -e "s|OUTPUTDIR|${OUTPUTDIR}|g" \
  -e "s|RUNNAME|${RUNNAME}|g" \
  -e "s|HOURS|${HOURS}|g" \
  -e "s|TRAIN|${TRAIN}|g" \
  -e "s|VALIDATION|${VALIDATION}|g" \
  -e "s|TEST|${TEST}|g" \
  <run_train_gpu.sh \
  >bash_files/run_${DATANAME}_${RUNNAME}_run0.sh
jid0=$(sbatch --parsable bash_files/run_${DATANAME}_${RUNNAME}_run0.sh)
echo $jid0
