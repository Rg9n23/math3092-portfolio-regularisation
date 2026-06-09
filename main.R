# Entry point for the regression modelling pipeline.
# Runs on one or more datasets defined in DATASETS (see 00_config.R).

source("R/00_config.R")
source("R/01_load_clean.R")
source("R/02_split.R")
source("R/10_fit_linear.R")
source("R/11_fit_regularized.R")
source("R/20_fit_tree.R")
source("R/30_fit_rf.R")
source("R/40_repeated_splits.R")
source("R/50_plots.R")
source("R/60_cross_dataset.R")
source("R/70_polynomial_complexity.R")
source("R/80_empirical_biasvar.R")
source("R/90_pca_signal.R")

check_packages()

# ---------------------------------------------------------------------------
# run_pipeline: run the full modelling pipeline for a single dataset config
# ---------------------------------------------------------------------------
run_pipeline <- function(ds_config) {
  ds_name  <- ds_config$name
  response <- ds_config$response

  # Dataset-specific output directories
  ds_fig_dir    <- file.path(FIG_DIR, ds_name)
  ds_output_dir <- file.path(OUTPUT_DIR, ds_name)
  if (!dir.exists(ds_fig_dir))    dir.create(ds_fig_dir, recursive = TRUE)
  if (!dir.exists(ds_output_dir)) dir.create(ds_output_dir, recursive = TRUE)

  message("\n=== Running pipeline: ", ds_config$label, " ===")
  message("  Data file: ", ds_config$file)
  message("  Response:  ", response)

  if (!file.exists(ds_config$file)) {
    stop("Dataset file not found: ", ds_config$file)
  }

  dat <- load_and_clean_data(
    path = ds_config$file,
    response = response,
    preprocess = ds_config$preprocess
  )
  message("  Loaded: ", nrow(dat), " rows, ", ncol(dat), " columns (",
          ncol(dat) - 1, " predictors)")

  split_obj <- split_train_test(dat, response = response, seed = SEED, train_prop = TRAIN_PROP)
  train <- split_obj$train
  test  <- split_obj$test

  # --- Eigenvalue analysis of X'X (for ridge/regularisation diagnostics) ---
  x_train_mat <- as.matrix(train[, setdiff(names(train), response), drop = FALSE])
  x_scaled <- scale(x_train_mat)
  sv <- svd(x_scaled)$d
  d_sq <- sv^2
  cum_var <- cumsum(d_sq) / sum(d_sq) * 100
  eigen_df <- data.frame(
    component = seq_along(sv),
    singular_value = sv,
    d_squared = d_sq,
    cumulative_var_pct = cum_var
  )
  readr::write_csv(eigen_df, file.path(ds_output_dir, "eigenvalue_analysis.csv"))
  kappa_val <- max(sv) / min(sv)
  message("  Condition number kappa(X): ", round(kappa_val, 1))

  # --- Fit models ---
  linear_res <- fit_linear_model(train, test, response = response)

  regularized_res <- fit_regularized_models(
    train, test,
    response = response,
    seed = SEED,
    fig_dir = ds_fig_dir,
    output_dir = ds_output_dir
  )

  tree_res <- fit_tree_model(
    train, test,
    response = response,
    fig_dir = ds_fig_dir,
    output_dir = ds_output_dir
  )

  # RF: set mtry sensibly for the dataset (default p/3, capped at p)
  p <- ncol(train) - 1
  rf_mtry <- max(1, min(RF_DEFAULTS$mtry, p))

  # Determine PDP features: use config if specified, otherwise fit once to find
  # top-3 by importance, then refit with PDPs
  rf_pdp <- ds_config$pdp_features
  if (is.null(rf_pdp)) {
    # Quick fit without artifacts to get variable importance
    rf_tmp <- fit_rf_model(
      train, test, response = response, seed = SEED,
      mtry = rf_mtry, min_node_size = RF_DEFAULTS$min_node_size,
      num_trees = RF_DEFAULTS$num_trees,
      save_artifacts = FALSE
    )
    rf_pdp <- names(sort(rf_tmp$extras$variable_importance, decreasing = TRUE))[1:3]
    message("  Auto-selected PDP features: ", paste(rf_pdp, collapse = ", "))
  }
  rf_pdp <- intersect(rf_pdp, setdiff(names(train), response))

  rf_res <- fit_rf_model(
    train, test,
    response = response,
    seed = SEED,
    mtry = rf_mtry,
    min_node_size = RF_DEFAULTS$min_node_size,
    num_trees = RF_DEFAULTS$num_trees,
    fig_dir = ds_fig_dir,
    output_dir = ds_output_dir,
    pdp_features = rf_pdp,
    pdp_2d = list(),
    save_artifacts = TRUE
  )

  # --- Build comparison table ---
  x_test <- as.matrix(test[, setdiff(names(test), response), drop = FALSE])
  y_test <- test[[response]]

  score_from_cv <- function(cv_fit, lambda_rule) {
    pred <- as.numeric(stats::predict(cv_fit, newx = x_test, s = lambda_rule))
    calc_metrics(y_test, pred)
  }

  fmt_param <- function(x, digits = 4) {
    if (!is.finite(x)) return("NA")
    if (abs(x) < 0.001) return(format(signif(x, 3), scientific = TRUE, trim = TRUE))
    format(round(x, digits), nsmall = digits, trim = TRUE)
  }

  lambdas <- regularized_res$extras$lambdas

  get_lambda_row <- function(lambdas, model_name) {
    row <- lambdas[lambdas$Model == model_name, , drop = FALSE]
    if (nrow(row) != 1) stop("Expected exactly 1 row for ", model_name)
    row
  }

  ridge_row <- get_lambda_row(lambdas, "Ridge")
  lasso_row <- get_lambda_row(lambdas, "Lasso")
  enet_row  <- get_lambda_row(lambdas, "Elastic Net")

  ridge_cv <- regularized_res$model$Ridge
  lasso_cv <- regularized_res$model$Lasso
  enet_cv  <- regularized_res$model$`Elastic Net`

  ridge_min_metrics <- score_from_cv(ridge_cv, "lambda.min")
  ridge_1se_metrics <- score_from_cv(ridge_cv, "lambda.1se")
  lasso_min_metrics <- score_from_cv(lasso_cv, "lambda.min")
  lasso_1se_metrics <- score_from_cv(lasso_cv, "lambda.1se")
  enet_min_metrics  <- score_from_cv(enet_cv,  "lambda.min")

  tree_cp <- tree_res$extras$cp_best
  rf_hp   <- rf_res$extras$hyperparameters

  model_table <- data.frame(
    model = c(
      "OLS",
      paste0("Ridge (lambda.min=", fmt_param(ridge_row$lambda_min), ")"),
      paste0("Ridge (lambda.1se=", fmt_param(ridge_row$lambda_1se), ")"),
      paste0("Lasso (lambda.min=", fmt_param(lasso_row$lambda_min), ")"),
      paste0("Lasso (lambda.1se=", fmt_param(lasso_row$lambda_1se), ")"),
      paste0("Elastic Net (alpha=", fmt_param(enet_row$alpha, 2),
             ", lambda.min=", fmt_param(enet_row$lambda_min), ")"),
      paste0("Decision Tree (pruned, cp=", fmt_param(tree_cp), ")"),
      paste0("Random Forest (mtry=", rf_hp$mtry,
             ", min.node.size=", rf_hp$min_node_size,
             ", num.trees=", rf_hp$num_trees, ")")
    ),
    test_rmse = c(
      unname(linear_res$metrics["RMSE"]),
      unname(ridge_min_metrics["RMSE"]), unname(ridge_1se_metrics["RMSE"]),
      unname(lasso_min_metrics["RMSE"]), unname(lasso_1se_metrics["RMSE"]),
      unname(enet_min_metrics["RMSE"]),
      unname(tree_res$metrics["RMSE"]),
      unname(rf_res$metrics["RMSE"])
    ),
    test_mae = c(
      unname(linear_res$metrics["MAE"]),
      unname(ridge_min_metrics["MAE"]), unname(ridge_1se_metrics["MAE"]),
      unname(lasso_min_metrics["MAE"]), unname(lasso_1se_metrics["MAE"]),
      unname(enet_min_metrics["MAE"]),
      unname(tree_res$metrics["MAE"]),
      unname(rf_res$metrics["MAE"])
    ),
    test_r2 = c(
      unname(linear_res$metrics["R2"]),
      unname(ridge_min_metrics["R2"]), unname(ridge_1se_metrics["R2"]),
      unname(lasso_min_metrics["R2"]), unname(lasso_1se_metrics["R2"]),
      unname(enet_min_metrics["R2"]),
      unname(tree_res$metrics["R2"]),
      unname(rf_res$metrics["R2"])
    ),
    row.names = NULL,
    check.names = FALSE
  )

  readr::write_csv(model_table, file.path(ds_output_dir, "model_table.csv"))

  plot_table <- data.frame(
    Model = model_table$model,
    RMSE = model_table$test_rmse,
    MAE = model_table$test_mae,
    R2 = model_table$test_r2,
    row.names = NULL,
    check.names = FALSE
  )

  # Repeated splits (if enabled)
  repeated_res <- NULL
  if (RUN_REPEATED_SPLITS) {
    is_large <- nrow(dat) > 10000
    is_heavy <- is_large || p > 50
    n_reps <- if (is_large) 0 else if (is_heavy) N_REPEATS_LARGE else N_REPEATS

    if (n_reps > 0) {
      repeated_res <- fit_repeated_splits(
        dat = dat,
        response = response,
        n_repeats = n_reps,
        base_seed = SEED,
        train_prop = TRAIN_PROP,
        num_trees = if (is_heavy) 300 else 500,
        fig_dir = ds_fig_dir,
        output_dir = ds_output_dir
      )
    } else {
      message("  Skipping repeated splits for large dataset (n=", nrow(dat), ", p=", p, ")")
    }
  }

  results <- list(
    linear = linear_res,
    regularized = regularized_res,
    tree = tree_res,
    rf = rf_res,
    model_table = plot_table
  )

  plot_res <- save_all_plots(
    train = train,
    test = test,
    results = results,
    repeated_results = repeated_res,
    response = response,
    response_label = ds_config$label,
    fig_dir = ds_fig_dir
  )

  # --- Polynomial complexity curve ---
  poly_res <- fit_polynomial_complexity(
    train, test, response = response,
    fig_dir = ds_fig_dir, output_dir = ds_output_dir
  )

  # --- PCA signal projection ---
  pca_res <- compute_pca_signal(
    train, test, response = response,
    fig_dir = ds_fig_dir, output_dir = ds_output_dir
  )

  message("  Saved model table: ", file.path(ds_output_dir, "model_table.csv"))
  message("  Saved plots:")
  for (f in plot_res$extras$files) message("   - ", f)

  list(
    name = ds_name,
    label = ds_config$label,
    dat = dat,
    train = train,
    test = test,
    results = results,
    model_table = model_table,
    repeated = repeated_res
  )
}

# ---------------------------------------------------------------------------
# Main: run pipeline on each configured dataset
# ---------------------------------------------------------------------------
ensure_dirs()

# Which datasets to run (default: all)
if (!exists("ACTIVE_DATASETS")) ACTIVE_DATASETS <- names(DATASETS)

all_results <- list()
for (ds_key in ACTIVE_DATASETS) {
  all_results[[ds_key]] <- run_pipeline(DATASETS[[ds_key]])
}

message("\n=== Pipeline complete for: ",
        paste(sapply(all_results, `[[`, "label"), collapse = ", "), " ===")

# ---------------------------------------------------------------------------
# Empirical bias-variance decomposition (selected datasets)
# ---------------------------------------------------------------------------
bv_datasets <- c("concrete", "communities_crime")
for (ds_key in intersect(bv_datasets, ACTIVE_DATASETS)) {
  ds <- DATASETS[[ds_key]]
  dat_bv <- load_and_clean_data(ds$file, ds$response, ds$preprocess)
  p_bv <- ncol(dat_bv) - 1
  is_heavy <- nrow(dat_bv) > 10000 || p_bv > 50
  n_reps_bv <- if (is_heavy) 15 else 30
  message("\n=== Empirical bias-variance: ", ds$label, " (", n_reps_bv, " splits) ===")
  compute_empirical_biasvar(
    dat_bv, response = ds$response, n_repeats = n_reps_bv,
    fig_dir = file.path(FIG_DIR, ds$name),
    output_dir = file.path(OUTPUT_DIR, ds$name)
  )
  rm(dat_bv)
}

# ---------------------------------------------------------------------------
# Cross-dataset synthesis (runs after all individual pipelines)
# ---------------------------------------------------------------------------
message("\n=== Running cross-dataset synthesis ===")
cross_results <- run_cross_dataset_synthesis(
  output_dir = OUTPUT_DIR,
  fig_dir = FIG_DIR,
  datasets = DATASETS
)
message("=== Cross-dataset synthesis complete ===")
