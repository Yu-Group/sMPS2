# Simplified MyProstateScore2.0 (sMPS2)

Code to reproduce results and figures in manuscript:

Tiffany M. Tang*, Yuping Zhang*, Ana M. Kenney, Cassie Xie,  Lanbo Xiao, Javed Siddiqui, Sudhir Srivastava, Martin G. Sanda, John T. Wei, Ziding Feng, Jeffrey J. Tosoian, Yingye Zheng, Arul M. Chinnaiyan&#8224;,  and Bin Yu&#8224;; for the EDRN-PCA3 Study Group. "A simplified MyProstateScore2.0 for high-grade prostate cancer". *Cancer Biomarkers*. 2025;42(1). [doi:10.1177/18758592241308755](https://journals.sagepub.com/doi/10.1177/18758592241308755)

Accompanying supplementary PCS documentation: https://yu-group.github.io/sMPS2/

In this work, we developed and rigorously stress-tested a novel non-invasive biomarker test, the simplified MyProstateScore2.0 (sMPS2), for accurate and early screening of clinically-significant prostate cancer. 

## Project Structure

- **[scripts/](./scripts/)**: contains all scripts to run sMPS model development and evaluation pipeline
	- Python scripts:
		- [01_train_prediction_methods.py](./scripts/01_train_prediction_methods.py): fits the prediction methods and estimates the feature importances from these learned models 
		- [02_aggregate_train_results.py](./scripts/02_aggregate_train_results.py): aggregates the training prediction and importance results across the many data splits, methods, and data preprocessing variants
		- [03_evaluate_prediction_methods.py](./scripts/03_evaluate_prediction_methods.py): evaluates the prediction methods on the test set given a particular choice of gene rankings and number of gene predictors
		- [04_aggregate_eval_results.py](./scripts/04_aggregate_eval_results.py): aggregates the test evaluation results across the many data splits, data preprocessing variants, gene ranking modes, and number of gene predictors
	- To run the full pipeline, the "driver" scripts should be run in sequential order, i.e., first [01_driver.sh](./scripts/01_driver_train_prediction_methods.sh), then [02_driver.sh](./scripts/02_driver_aggregate_train_results.sh), then [03_driver.sh](./scripts/03_driver_evaluate_prediction_methods.sh), then [04_driver.sh](./scripts/04_driver_aggregate_eval_results.sh)
- **[model_config/](./model_config/)**: model configuration file, specifying the models, hyperparameters, and feature importances under study
- **[functions/](./functions/)**: contains all functions necessary to reproduce sMPS model development pipeline
- **[notebooks/](./notebooks/)**: contains files to reproduce [supplementary PCS documentation](https://yu-group.github.io/sMPS2/)

## Citation

To cite sMPS2, please use:

```
@article{tang2025simplified,
  author = {Tiffany M Tang and Yuping Zhang and Ana M Kenney and Cassie Xie and Lanbo Xiao and Javed Siddiqui and Sudhir Srivastava and Martin G Sanda and John T Wei and Ziding Feng and Jeffrey J Tosoian and Yingye Zheng and Arul M Chinnaiyan and Bin Yu},
  title ={A simplified MyProstateScore2.0 for high-grade prostate cancer},
  journal = {Cancer Biomarkers},
  volume = {42},
  number = {1},
  pages = {18758592241308755},
  year = {2025},
  doi = {10.1177/18758592241308755},
  note = {PMID: 40109218},
  URL = {https://doi.org/10.1177/18758592241308755}
}
```