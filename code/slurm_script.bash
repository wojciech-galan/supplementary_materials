#!/usr/bin/bash
#SBATCH -A hostphylumpred
#SBATCH -p plgrid-short
#SBATCH -N 1
#SBATCH --ntasks-per-node=24

mut_probas = 0.05 0.075 0.1 0.125 0.15 0.175 0.2

cd wirusy/supplementary_materials/code/
module load plgrid/tools/python/2.7.13
for mut_proba in $mut_probas
do
    python knn_ga_feature_selection.py 10 5 cxUniform $mut_proba 2 1 3 100 200 ../ga/ --cpus 24
done