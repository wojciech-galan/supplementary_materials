#!/usr/bin/bash
#SBATCH -A hostphylumpred
#SBATCH -p plgrid
#SBATCH --time=3-00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=8GB

outdir="bottom_up_feature_selection_results_qda"
infile="$SCRATCHDIR/datasets/splits.dump"
mkdir $SCRATCHDIR/datasets
mkdir -p $SCRATCHDIR/$outdir
mkdir -p $SLURM_SUBMIT_DIR/$outdir
cp $SLURM_SUBMIT_DIR/datasets/splits.dump  $SCRATCHDIR/datasets
cp $SLURM_SUBMIT_DIR/$outdir $SCRATCHDIR/ -r
echo $SCRATCHDIR

module load plgrid/tools/python/2.7.13

python code/feature_selection_bottom_up.py $SCRATCHDIR/$outdir 2 --infile $infile
