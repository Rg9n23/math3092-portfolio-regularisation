# MATH3092 - Regularisation and Model Complexity in Predictive Modelling

Mathematics project (University of Southampton) investigating how regularisation
and model complexity affect predictive accuracy across multiple regression datasets.

## Models

- OLS baseline
- Ridge, Lasso, and Elastic Net regression (glmnet, 10-fold CV)
- Decision tree with cost-complexity pruning (rpart, 1-SE rule)
- Random forest (ranger, permutation importance + partial dependence)

## Datasets

Seven UCI regression datasets are run through the same pipeline:
Concrete, Airfoil, Bike Sharing, Communities & Crime, Energy Efficiency,
Superconductor, and Wine Quality.

## Structure

```
main.R           Entry point — sources all R/ scripts, runs each dataset
R/
  00_config.R    Global config, helper functions (RMSE, MAE, R²)
  01_load_clean.R    Data loading and cleaning
  02_split.R         Train/test split (80/20)
  10_fit_linear.R    OLS
  11_fit_regularized.R   Ridge / Lasso / Elastic Net
  20_fit_tree.R      Decision tree (unpruned + pruned)
  30_fit_rf.R        Random forest
  40_repeated_splits.R   Robustness over 50 random splits
  50_plots.R         Predicted-vs-actual, RMSE bar, OLS diagnostics
  51_shrinkage_comparison.R   Ridge vs Lasso coefficient paths
  60_cross_dataset.R     Multi-dataset comparison and synthesis
  70_polynomial_complexity.R  Polynomial complexity curve
  80_empirical_biasvar.R      Empirical bias-variance decomposition
  90_pca_signal.R             PCA signal alignment
Report/main.pdf  Compiled report
```

## Running

```r
source("main.R")
```

Requires R with packages: glmnet, rpart, rpart.plot, ranger, pdp, readr.
Datasets should be placed in `data/`.
