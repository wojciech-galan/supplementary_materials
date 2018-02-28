#!/usr/bin/bash
#SBATCH -A hostphylumpred
#SBATCH -p plgrid
#SBATCH --time=3-00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=3GB

mut_probas='0.15 0.175 0.2 0.25 0.3'
crossovers='cxUniform'

subdir="ga_res/qda/$1"
echo "resuls written to $subdir"
infile="$SCRATCHDIR/datasets/splits.dump"
mkdir $SCRATCHDIR/datasets
mkdir -p $SCRATCHDIR/$subdir
mkdir -p $SLURM_SUBMIT_DIR/$subdir
cp $SLURM_SUBMIT_DIR/datasets/splits.dump $SCRATCHDIR/datasets

module load plgrid/tools/python/2.7.13
for mean in {20..60..10}
do
    for crossover in $crossovers
    do
        for mut_proba in $mut_probas
        do
            for tournsize in {2..6..2}
            do
                python code/qda_ga_feature_selection.py $mean 5 $crossover $mut_proba $tournsize 1 500 200 $SCRATCHDIR/$subdir --infile $infile --cpus 6 --searchdir $SLURM_SUBMIT_DIR/$subdir
                mv $SCRATCHDIR/$subdir/* $SLURM_SUBMIT_DIR/$subdir
            done
        done
    done
done
