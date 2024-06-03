#!/bin/bash
#SBATCH --time=1-00:00:00
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#
#SBATCH -a 1-10

RESULTSPATH="../results"
ks=(1 2 3 4 5 6 7 10 12 15 17 20 25 30 40 55)

for k in "${ks[@]}"
do
    echo ${k}
    python3 03_evaluate_prediction_methods.py \
      --split_seed ${SLURM_ARRAY_TASK_ID} \
      --results_path ${RESULTSPATH}"/"${1} \
      --topk ${k} \
      --topk_mode ${2} \
      --include_clin \
      --scale_X
done