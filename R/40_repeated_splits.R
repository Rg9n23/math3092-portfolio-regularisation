# Repeated split evaluation for performance stability.

fit_repeated_splits <- function(
    dat,
    response = RESPONSE,
    n_repeats = N_REPEATS,
    base_seed = SEED,
    train_prop = TRAIN_PROP,
    nfolds_regularized = 5,
    alpha_grid = seq(0.1, 0.9, by = 0.1),
    stratify = FALSE,
    num_trees = RF_DEFAULTS$num_trees,
    fig_dir = FIG_DIR,
    output_dir = OUTPUT_DIR
) {
  assert_response(dat, response)
  
  if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  metric_row <- function(repeat_id, seed_value, model_name, metrics) {
    data.frame(
      Repeat = repeat_id,
      Seed = seed_value,
      Model = model_name,
      RMSE = unname(metrics["RMSE"]),
      MAE  = unname(metrics["MAE"]),
      R2   = unname(metrics["R2"]),
      row.names = NULL,
      check.names = FALSE
    )
  }
  
  fit_cv_glmnet <- function(tr, te, alpha, seed_value) {
    x_train <- as.matrix(tr[, setdiff(names(tr), response), drop = FALSE])
    y_train <- tr[[response]]
    x_test  <- as.matrix(te[, setdiff(names(te), response), drop = FALSE])
    y_test  <- te[[response]]
    
    set.seed(seed_value)
    cv_fit <- glmnet::cv.glmnet(
      x = x_train,
      y = y_train,
      alpha = alpha,
      nfolds = nfolds_regularized,
      standardize = TRUE,
      type.measure = "mse"
    )
    
    pred <- as.numeric(stats::predict(cv_fit, newx = x_test, s = "lambda.min"))
    list(
      model = cv_fit,
      pred = pred,
      metrics = calc_metrics(y_test, pred),
      lambda_min = cv_fit$lambda.min,
      lambda_1se = cv_fit$lambda.1se
    )
  }
  
  fit_cv_elastic_net <- function(tr, te, alpha_grid, seed_value) {
    x_train <- as.matrix(tr[, setdiff(names(tr), response), drop = FALSE])
    y_train <- tr[[response]]
    x_test  <- as.matrix(te[, setdiff(names(te), response), drop = FALSE])
    y_test  <- te[[response]]
    
    # choose alpha using training CV error at lambda.min (no test leakage)
    cv_list <- vector("list", length(alpha_grid))
    cv_score <- numeric(length(alpha_grid))
    
    for (i in seq_along(alpha_grid)) {
      set.seed(seed_value) # keep folds comparable across alphas
      cv_fit <- glmnet::cv.glmnet(
        x = x_train,
        y = y_train,
        alpha = alpha_grid[i],
        nfolds = nfolds_regularized,
        standardize = TRUE,
        type.measure = "mse"
      )
      cv_list[[i]] <- cv_fit
      
      # CV MSE at lambda.min
      idx_min <- which(cv_fit$lambda == cv_fit$lambda.min)[1]
      cv_score[i] <- cv_fit$cvm[idx_min]
    }
    
    best_i <- which.min(cv_score)
    best_alpha <- alpha_grid[best_i]
    best_cv <- cv_list[[best_i]]
    
    pred <- as.numeric(stats::predict(best_cv, newx = x_test, s = "lambda.min"))
    list(
      model = best_cv,
      pred = pred,
      metrics = calc_metrics(y_test, pred),
      alpha = best_alpha,
      lambda_min = best_cv$lambda.min,
      lambda_1se = best_cv$lambda.1se
    )
  }
  
  fit_pruned_tree_once <- function(tr, te, seed_value) {
    fml <- stats::as.formula(paste(response, "~ ."))
    
    set.seed(seed_value)
    tree_unpruned <- rpart::rpart(
      formula = fml,
      data = tr,
      method = "anova",
      control = rpart::rpart.control(
        cp = 1e-04,
        minsplit = 20,
        maxdepth = 10,
        xval = 10
      )
    )
    
    cp_table <- tree_unpruned$cptable
    best_row <- which.min(cp_table[, "xerror"])
    xerr_min <- cp_table[best_row, "xerror"]
    xstd_min <- cp_table[best_row, "xstd"]
    thresh <- xerr_min + xstd_min
    
    eligible <- which(cp_table[, "xerror"] <= thresh)
    cp_1se <- cp_table[min(eligible), "CP"]
    cp_best <- cp_1se
    
    tree_pruned <- rpart::prune(tree_unpruned, cp = cp_best)
    pred <- as.numeric(stats::predict(tree_pruned, newdata = te))
    
    list(
      model = tree_pruned,
      pred = pred,
      metrics = calc_metrics(te[[response]], pred),
      cp_best = cp_best
    )
  }
  
  rows <- vector("list", n_repeats * 6L)
  row_idx <- 1L
  
  for (r in seq_len(n_repeats)) {
    seed_r <- base_seed + r
    
    split_r <- split_train_test(
      dat = dat,
      response = response,
      seed = seed_r,
      train_prop = train_prop,
      stratify = stratify
    )
    tr <- split_r$train
    te <- split_r$test
    
    # OLS
    ols_res <- fit_linear_model(tr, te, response = response)
    
    # Regularised (separate seeds so CV randomness is stable per repeat)
    ridge_res <- fit_cv_glmnet(tr, te, alpha = 0, seed_value = seed_r + 10000)
    lasso_res <- fit_cv_glmnet(tr, te, alpha = 1, seed_value = seed_r + 20000)
    enet_res  <- fit_cv_elastic_net(tr, te, alpha_grid = alpha_grid, seed_value = seed_r + 30000)
    
    # Tree
    tree_res <- fit_pruned_tree_once(tr, te, seed_value = seed_r + 40000)
    
    # RF (fixed hyperparams; no artifacts)
    rf_res <- fit_rf_model(
      tr, te,
      response = response,
      seed = seed_r,
      mtry = RF_DEFAULTS$mtry,
      min_node_size = RF_DEFAULTS$min_node_size,
      num_trees = num_trees,
      save_artifacts = FALSE
    )
    
    rows[[row_idx]] <- metric_row(r, seed_r, "OLS", ols_res$metrics); row_idx <- row_idx + 1L
    rows[[row_idx]] <- metric_row(r, seed_r, "Ridge", ridge_res$metrics); row_idx <- row_idx + 1L
    rows[[row_idx]] <- metric_row(r, seed_r, "Lasso", lasso_res$metrics); row_idx <- row_idx + 1L
    rows[[row_idx]] <- metric_row(r, seed_r, "Elastic Net", enet_res$metrics); row_idx <- row_idx + 1L
    rows[[row_idx]] <- metric_row(r, seed_r, "Decision Tree (Pruned)", tree_res$metrics); row_idx <- row_idx + 1L
    rows[[row_idx]] <- metric_row(r, seed_r, "Random Forest", rf_res$metrics); row_idx <- row_idx + 1L

    rm(ols_res, ridge_res, lasso_res, enet_res, tree_res, rf_res, tr, te, split_r)
    gc(verbose = FALSE)
  }
  
  metrics_long <- do.call(rbind, rows)
  
  csv_path <- file.path(output_dir, "repeated_splits_results.csv")
  readr::write_csv(metrics_long, csv_path)
  
  # Summary table (mean/sd) robustly
  models <- sort(unique(metrics_long$Model))
  summary <- data.frame(Model = models, row.names = NULL)
  
  summarise_metric <- function(df, metric) {
    out_mean <- tapply(df[[metric]], df$Model, mean)
    out_sd   <- tapply(df[[metric]], df$Model, stats::sd)
    data.frame(
      Model = names(out_mean),
      mean = as.numeric(out_mean),
      sd = as.numeric(out_sd),
      row.names = NULL
    )
  }
  
  rmse_sum <- summarise_metric(metrics_long, "RMSE")
  mae_sum  <- summarise_metric(metrics_long, "MAE")
  r2_sum   <- summarise_metric(metrics_long, "R2")
  
  summary <- merge(summary, rmse_sum, by = "Model", all.x = TRUE)
  names(summary)[names(summary) %in% c("mean","sd")] <- c("RMSE_mean","RMSE_sd")
  summary <- merge(summary, mae_sum, by = "Model", all.x = TRUE)
  names(summary)[names(summary) %in% c("mean","sd")] <- c("MAE_mean","MAE_sd")
  summary <- merge(summary, r2_sum, by = "Model", all.x = TRUE)
  names(summary)[names(summary) %in% c("mean","sd")] <- c("R2_mean","R2_sd")
  
  # Win-rate: % of repeats where model has lowest RMSE
  best_by_repeat <- by(metrics_long, metrics_long$Repeat, function(d) d$Model[which.min(d$RMSE)][1])
  best_by_repeat <- as.character(best_by_repeat)
  win_rate <- sort(table(best_by_repeat) / length(best_by_repeat), decreasing = TRUE)
  win_rate_df <- data.frame(Model = names(win_rate), win_rate = as.numeric(win_rate), row.names = NULL)
  
  readr::write_csv(summary, file.path(output_dir, "repeated_splits_summary.csv"))
  readr::write_csv(win_rate_df, file.path(output_dir, "repeated_splits_win_rate.csv"))
  
  # Plots
  rmse_fig <- file.path(fig_dir, "repeated_rmse_boxplot.pdf")
  grDevices::pdf(rmse_fig, width = 10, height = 7, pointsize = 10)
  op <- par(mar = c(8, 5, 4, 2))
  boxplot(RMSE ~ Model, data = metrics_long, las = 2, ylab = "RMSE",
          main = sprintf("RMSE across %d repeated 80/20 splits", n_repeats))
  par(op)
  grDevices::dev.off()

  mae_fig <- file.path(fig_dir, "repeated_mae_boxplot.pdf")
  grDevices::pdf(mae_fig, width = 10, height = 7, pointsize = 10)
  op <- par(mar = c(8, 5, 4, 2))
  boxplot(MAE ~ Model, data = metrics_long, las = 2, ylab = "MAE",
          main = sprintf("MAE across %d repeated 80/20 splits", n_repeats))
  par(op)
  grDevices::dev.off()
  
  list(
    model = NULL,
    pred = NULL,
    metrics = metrics_long,
    extras = list(
      summary = summary,
      win_rate = win_rate_df,
      n_repeats = n_repeats,
      base_seed = base_seed,
      output_csv = csv_path,
      figures = c(
        repeated_rmse_boxplot = rmse_fig,
        repeated_mae_boxplot = mae_fig
      )
    )
  )
}