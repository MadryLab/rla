#!/bin/bash

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

if [[ ! -d bash_files ]];
then
  mkdir bash_files
fi


MODELDIR=$1
DATASET=$2
DATANAME=${2##*/}
OUTPUTDIR=$3
SUBSET=$4
if [[ ! -d $OUTPUTDIR ]];
then
  mkdir $OUTPUTDIR
fi
RUNNAME=${3##*/}
sed \
  -e "s|DATASET|${DATASET}|g" \
  -e "s|DATANAME|${DATANAME}|g" \
  -e "s|MODELDIR|${MODELDIR}|g" \
  -e "s|OUTPUTDIR|${OUTPUTDIR}|g" \
  -e "s|RUNNAME|${RUNNAME}|g" \
  -e "s|SUBSET|${SUBSET}|g" \
  <run_eval.sh \
  >bash_files/eval_${DATANAME}_${RUNNAME}.sh
jid0=$(sbatch --parsable bash_files/eval_${DATANAME}_${RUNNAME}.sh)
echo $jid0
