#!/bin/bash
#SBATCH -N 1
#SBATCH --time=06:00:00
#SBATCH -n 16
#SBATCH --mem=0
#SBATCH -o logfiles/compress-files-output-RUNNAME.out
#SBATCH -e logfiles/compress-files-error-RUNNAME.out

tar -cvzf OUTPUTDIR/etabs.tar.gz OUTPUTDIR/etabs/
rm -rf OUTPUTDIR/etabs/
