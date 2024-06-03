#!/bin/bash

sbatch --job-name=train_original 01_train_prediction_methods.sh original
sbatch --job-name=train_no_sample_filtering 01_train_prediction_methods.sh no_sample_filtering
sbatch --job-name=train_ct40 01_train_prediction_methods.sh ct40
sbatch --job-name=train_normct21 01_train_prediction_methods.sh normct21
