#!/bin/bash

# evaluate models - method-specific
sbatch --job-name="eval_naive_original" 03_evaluate_prediction_methods.sh original naive
sbatch --job-name="eval_naive_no_sample_filtering" 03_evaluate_prediction_methods.sh no_sample_filtering naive
sbatch --job-name="eval_naive_ct40" 03_evaluate_prediction_methods.sh ct40 naive
sbatch --job-name="eval_naive_normct21" 03_evaluate_prediction_methods.sh normct21 naive

# evaluate models - model-ensembled
sbatch --job-name="eval_ensembled_original" 03_evaluate_prediction_methods.sh original ensemble
sbatch --job-name="eval_ensembled_no_sample_filtering" 03_evaluate_prediction_methods.sh no_sample_filtering ensemble
sbatch --job-name="eval_ensembled_ct40" 03_evaluate_prediction_methods.sh ct40 ensemble
sbatch --job-name="eval_ensembled_normct21" 03_evaluate_prediction_methods.sh normct21 ensemble

# evaluate models - pcs-ensembled
sbatch --job-name="eval_pcs_original" 03_evaluate_prediction_methods.sh original pcs
sbatch --job-name="eval_pcs_no_sample_filtering" 03_evaluate_prediction_methods.sh no_sample_filtering pcs
sbatch --job-name="eval_pcs_ct40" 03_evaluate_prediction_methods.sh ct40 pcs
sbatch --job-name="eval_pcs_normct21" 03_evaluate_prediction_methods.sh normct21 pcs