#!/usr/bin/bash
#SBATCH -A hostphylumpred
#SBATCH -p plgrid
#SBATCH --time=3-00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=3GB

mut_probas='0.05 0.1 0.15 0.2 0.25 0.3'
crossovers='cxUniform'

subdir="ga_res/knn/$1"
echo "resuls written to $subdir"
infile="$SCRATCHDIR/datasets/splits.dump"
mkdir $SCRATCHDIR/datasets
mkdir -p $SCRATCHDIR/$subdir
mkdir -p $SLURM_SUBMIT_DIR/$subdir
cp $SLURM_SUBMIT_DIR/datasets/splits.dump $SCRATCHDIR/datasets

module load plgrid/tools/python/2.7.13
for mean in {10..20..10}
do
    for crossover in $crossovers
    do
        for mut_proba in $mut_probas
        do
            for tournsize in {4..8..2}
            do
                for k in {9..9}
                do
                    python code/knn_ga_feature_selection.py $mean 5 $crossover $mut_proba $tournsize 1 $k 500 200 $SCRATCHDIR/$subdir --infile $infile --cpus 6 --searchdir $SLURM_SUBMIT_DIR/$subdir
                done
                mv $SCRATCHDIR/$subdir/* $SLURM_SUBMIT_DIR/$subdir
            done
        done
    done
done