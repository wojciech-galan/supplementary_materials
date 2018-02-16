#!/usr/bin/bash
#SBATCH -A hostphylumpred
#SBATCH -p plgrid
#SBATCH --time=3-00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=3GB
module load plgrid/tools/python/2.7.13
module load plgrid/apps/r/3.3.0

export R_LIBS_USER="/net/people/plgalan/R/libs"

subdir="svm_res"
infile="$SCRATCHDIR/datasets/splits.dump"
outdir="$SCRATCHDIR/$subdir"

mkdir $SCRATCHDIR/datasets
mkdir -p $SCRATCHDIR/$subdir
mkdir -p $SLURM_SUBMIT_DIR/$subdir
cp $SLURM_SUBMIT_DIR/datasets/splits.dump $SCRATCHDIR/datasets

python code/feature_selection_for_svc_penalizedSVM.py --infile $infile --outdir $outdir
#for mean in {10..90..10}
#do
#    for crossover in $crossovers
#    do
#        for mut_proba in $mut_probas
#        do
#            for tournsize in {2..8..2}
#            do
#                for k in {1..7..2}
#                do
#                    python code/knn_ga_feature_selection.py $mean 5 $crossover $mut_proba $tournsize 1 $k 100 200 $SCRATCHDIR/$subdir --infile $infile --cpus 6 --searchdir $SLURM_SUBMIT_DIR/$subdir
#                done
#                mv $SCRATCHDIR/$subdir/* $SLURM_SUBMIT_DIR/$subdir
#            done
#        done
#    done
