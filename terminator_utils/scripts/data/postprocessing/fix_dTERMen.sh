#!/bin/bash
#SBATCH -N 1
#SBATCH --mem=8000
#SBATCH --partition=defq
#SBATCH --time=1-00:00:00
#SBATCH -o OUTPUTDIR/ID/design.oFIXED
#SBATCH -e OUTPUTDIR/ID/design.eFIXED

# compute what directory this file is in
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
# DIR is the directory this file is in, e.g. postprocessing
$DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

# collect MST_PATH from config file
cwd=$PWD
cd $DIR
cd ../../
source config.sh
cd $cwd

$MST_PATH/bin/design --p ID.pdb --o ID --c $DIR/design.configfile
