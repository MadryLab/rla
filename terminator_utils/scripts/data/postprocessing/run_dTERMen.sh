#!/bin/bash
#SBATCH -N 1
#SBATCH --mem=2000
#SBATCH --partition=xeon-p8
#SBATCH --time=0:30:00
#SBATCH -o OUTPUTDIR/ID-output.out
#SBATCH -e OUTPUTDIR/ID-error.err
# collect MST_PATH from config file

cwd=$PWD
cd POSTDIR
source ../../config.sh
cd $cwd
$MST_PATH/bin/design --p ID.pdb --o ID --c POSTDIR/design.configfile

