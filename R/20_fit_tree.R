# Regression tree (CART).

fit_tree_model <- function(
    train,
    test,
    response = RESPONSE,
    cp = 1e-04,
    minsplit = 20,
    maxdepth = 10,
    xval = 10,
    use_1se = TRUE,
    fig_dir = FIG_DIR,
    output_dir = OUTPUT_DIR
) {
  assert_response(train, response)
  assert_response(test, response)
  
  if (!requireNamespace("rpart.plot", quietly = TRUE)) {
    stop("Package 'rpart.plot' is required for tree plotting.")
  }
  if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  fml <- stats::as.formula(paste(response, "~ ."))
  
  unpruned_model <- rpart::rpart(
    formula = fml,
    data = train,
    method = "anova",
    control = rpart::rpart.control(
      cp = cp,
      minsplit = minsplit,
      maxdepth = maxdepth,
      xval = xval
    )
  )
  
  # Metrics: train + test for unpruned
  pred_unpruned_train <- as.numeric(stats::predict(unpruned_model, newdata = train))
  pred_unpruned_test  <- as.numeric(stats::predict(unpruned_model, newdata = test))
  metrics_unpruned_train <- calc_metrics(train[[response]], pred_unpruned_train)
  metrics_unpruned_test  <- calc_metrics(test[[response]],  pred_unpruned_test)
  
  # Save unpruned tree diagram
  path_unpruned <- file.path(fig_dir, "decision_tree_unpruned.pdf")
  grDevices::pdf(path_unpruned, width = 11, height = 7.5, pointsize = 10)
  rpart.plot::rpart.plot(
    unpruned_model,
    type = 2,
    extra = 101,
    under = TRUE,
    faclen = 0,
    tweak = 1.0,
    compress = TRUE,
    clip.right.labs = FALSE,
    fallen.leaves = TRUE,
    roundint = FALSE,
    main = "Unpruned regression tree"
  )
  grDevices::dev.off()
  
  # Pruning selection
  cp_table <- unpruned_model$cptable
  best_row <- which.min(cp_table[, "xerror"])
  xerr_min <- cp_table[best_row, "xerror"]
  xstd_min <- cp_table[best_row, "xstd"]
  cp_min_xerror <- cp_table[best_row, "CP"]
  
  # 1-SE rule: choose simplest tree (largest CP) with xerror <= min + xstd_at_min
  thresh <- xerr_min + xstd_min
  eligible <- which(cp_table[, "xerror"] <= thresh)
  cp_1se <- cp_table[min(eligible), "CP"]
  
  cp_best <- if (use_1se) cp_1se else cp_min_xerror
  
  # Save pruning curve
  path_cp <- file.path(fig_dir, "tree_cp_curve.pdf")
  grDevices::pdf(path_cp, width = 10, height = 7, pointsize = 10)
  plot(
    log(cp_table[, "CP"]),
    cp_table[, "xerror"],
    type = "b",
    pch = 16,
    xlab = "log(CP)",
    ylab = "Cross-validated relative error (xerror)",
    main = "Tree pruning curve (from cptable)"
  )
  arrows(
    x0 = log(cp_table[, "CP"]),
    y0 = cp_table[, "xerror"] - cp_table[, "xstd"],
    x1 = log(cp_table[, "CP"]),
    y1 = cp_table[, "xerror"] + cp_table[, "xstd"],
    angle = 90,
    code = 3,
    length = 0.03
  )
  graphics::abline(v = log(cp_min_xerror), lty = 2, lwd = 2)
  graphics::abline(v = log(cp_1se), lty = 3, lwd = 2)
  graphics::abline(v = log(cp_best), lty = 1, lwd = 2)
  graphics::legend(
    "topright",
    legend = c(
      sprintf("cp_min_xerror = %.6g", cp_min_xerror),
      sprintf("cp_1se = %.6g", cp_1se),
      sprintf("cp_used = %.6g", cp_best)
    ),
    lty = c(2, 3, 1),
    lwd = 2,
    bty = "n"
  )
  grDevices::dev.off()
  
  # Fit pruned model (chosen rule)
  pruned_model <- rpart::prune(unpruned_model, cp = cp_best)
  
  pred_pruned_train <- as.numeric(stats::predict(pruned_model, newdata = train))
  pred_pruned_test  <- as.numeric(stats::predict(pruned_model, newdata = test))
  metrics_pruned_train <- calc_metrics(train[[response]], pred_pruned_train)
  metrics_pruned_test  <- calc_metrics(test[[response]],  pred_pruned_test)
  
  # Save pruned tree diagram
  path_pruned <- file.path(fig_dir, "decision_tree_pruned.pdf")
  grDevices::pdf(path_pruned, width = 11, height = 7.5, pointsize = 10)
  rpart.plot::rpart.plot(
    pruned_model,
    type = 2,
    extra = 101,
    under = TRUE,
    faclen = 0,
    tweak = 1.05,
    compress = TRUE,
    clip.right.labs = FALSE,
    fallen.leaves = TRUE,
    roundint = FALSE,
    main = if (use_1se) "Pruned tree (1-SE rule)" else "Pruned tree (min CV error)"
  )
  grDevices::dev.off()
  
  # Save results table (train + test)
  tree_results <- data.frame(
    Model = c("Tree (Unpruned)", "Tree (Pruned)"),
    RMSE_train = c(unname(metrics_unpruned_train["RMSE"]), unname(metrics_pruned_train["RMSE"])),
    RMSE_test  = c(unname(metrics_unpruned_test["RMSE"]),  unname(metrics_pruned_test["RMSE"])),
    MAE_train  = c(unname(metrics_unpruned_train["MAE"]),  unname(metrics_pruned_train["MAE"])),
    MAE_test   = c(unname(metrics_unpruned_test["MAE"]),   unname(metrics_pruned_test["MAE"])),
    R2_train   = c(unname(metrics_unpruned_train["R2"]),   unname(metrics_pruned_train["R2"])),
    R2_test    = c(unname(metrics_unpruned_test["R2"]),    unname(metrics_pruned_test["R2"])),
    cp_used    = c(NA_real_, cp_best),
    row.names = NULL
  )
  readr::write_csv(tree_results, file.path(output_dir, "tree_results.csv"))
  
  list(
    model = pruned_model,
    pred = pred_pruned_test,
    metrics = metrics_pruned_test,
    extras = list(
      unpruned_model = unpruned_model,
      unpruned_train_metrics = metrics_unpruned_train,
      unpruned_test_metrics = metrics_unpruned_test,
      pruned_train_metrics = metrics_pruned_train,
      pruned_test_metrics = metrics_pruned_test,
      cp_table = cp_table,
      cp_min_xerror = cp_min_xerror,
      cp_1se = cp_1se,
      cp_best = cp_best,
      use_1se = use_1se,
      tree_results = tree_results,
      variable_importance_unpruned = unpruned_model$variable.importance,
      variable_importance = pruned_model$variable.importance,
      figures = c(
        decision_tree_unpruned = path_unpruned,
        tree_cp_curve = path_cp,
        decision_tree_pruned = path_pruned
      )
    )
  )
}