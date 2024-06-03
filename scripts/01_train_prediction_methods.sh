#!/bin/bash
#SBATCH --time=1:00:00
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#
#SBATCH -a 1-10

module load python/3.7

RESULTSPATH="../results"
DATAPATH="../data"

python3 01_train_prediction_methods.py \
  --data_fpath ${DATAPATH}"/data_"${1}".csv" \
  --data_index \
  --split_seed ${SLURM_ARRAY_TASK_ID} \
  --results_path ${RESULTSPATH}"/"${1} \
  --include_clin \
  --scale_X
