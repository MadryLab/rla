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

echo $DIR
cd $DIR

# check if shell_scripts exists and if not create it
if [[ ! -d shell_scripts ]];
then
  mkdir shell_scripts
fi

# PDBDIR=$1
# PDBROOT=$2
OUTPUTDIR=$1
RUNNAME=${1##*/}
# echo $PDBDIR
echo $OUTPUTDIR
echo $RUNNAME
sed \
    -e "s|OUTPUTDIR|${OUTPUTDIR}|g" \
    <to_etab_only.sh \
    >shell_scripts/to_etab_only_${RUNNAME}.sh
sbatch shell_scripts/to_etab_only_${RUNNAME}.sh
