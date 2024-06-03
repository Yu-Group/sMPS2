#!/bin/bash

module load python/3.7

RESULTSPATH="../results"
DATAPATH="../data"
PIPELINES="original,no_sample_filtering,ct40,normct21"
SPLITSEEDS="1,2,3,4,5,6,7,8,9,10"

## naive results

python3 04_aggregate_eval_results.py \
    --split_seeds ${SPLITSEEDS} \
    --results_root_path ${RESULTSPATH} \
    --results_dirs ${PIPELINES} \
    --include_clin \
    --scale_X \
    --topk_mode naive

## ensemble results

python3 04_aggregate_eval_results.py \
    --split_seeds ${SPLITSEEDS} \
    --results_root_path ${RESULTSPATH} \
    --results_dirs ${PIPELINES} \
    --include_clin \
    --scale_X \
    --topk_mode ensemble

## pcs results

python3 04_aggregate_eval_results.py \
    --split_seeds ${SPLITSEEDS} \
    --results_root_path ${RESULTSPATH} \
    --results_dirs ${PIPELINES} \
    --include_clin \
    --scale_X \
    --topk_mode pcs
