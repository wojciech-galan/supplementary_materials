#!/usr/bin/bash
#SBATCH -A hostphylumpred
#SBATCH -p plgrid-short
#SBATCH -N 1
#SBATCH --ntasks-per-node=24
cd wirusy/supplementary_materials/code/
module load plgrid/tools/python/2.7.13
python knn_ga_feature_selection.py 10 5 cxUniform 0.1 2 1 3 20 2 ../ga/ --cpus 24