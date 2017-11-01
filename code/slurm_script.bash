#!/usr/bin/bash
#SBATCH -A hostphylumpred
#SBATCH -p plgrid
#SBATCH --time=3-00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=12GB

mut_probas='0.05 0.075 0.1 0.125 0.15 0.175 0.2'
crossovers='cxOnePoint cxTwoPoint cxUniform'

subdir="ga_res/knn/1"
infile="$SCRATCHDIR/datasets/splits.dump"
mkdir $SCRATCHDIR/datasets
mkdir -p $SCRATCHDIR/$subdir
mkdir -p $SLURM_SUBMIT_DIR/$subdir
cp $SLURM_SUBMIT_DIR/datasets/splits.dump $SCRATCHDIR/datasets

module load plgrid/tools/python/2.7.13
for mean in {10..90..10}
do
    for crossover in $crossovers
    do
        for mut_proba in $mut_probas
        do
            for tournsize in {2..8..2}
            do
                for k in {1..7..2}
                do
                    python knn_ga_feature_selection.py $mean 5 $crossover $mut_proba $tournsize 1 $k 100 200 $SCRATCHDIR/$subdir --infile $infile --cpus 6
                done
                mv $SCRATCHDIR/$subdir/* SLURM_SUBMIT_DIR/$subdir
            done
        done
    done
done