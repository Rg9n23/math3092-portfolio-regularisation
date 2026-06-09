# Empirical bias-variance decomposition via repeated splits.
# For each test point across splits, computes Bias^2 and Var of predictions.

compute_empirical_biasvar <- function(
    dat,
    response,
    n_repeats = 30,
    base_seed = SEED,
    train_prop = TRAIN_PROP,
    num_trees = 500,
    fig_dir = FIG_DIR,
    output_dir = OUTPUT_DIR
) {
  assert_response(dat, response)
  if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  # Reset row names so indices are 1:n (preprocessing may have dropped rows)
  rownames(dat) <- NULL
  n <- nrow(dat)
  p <- ncol(dat) - 1
  y <- dat[[response]]
  predictors <- setdiff(names(dat), response)

  # Storage: list of prediction vectors per model, indexed by observation
  model_names <- c("OLS", "Ridge", "Lasso", "Decision Tree", "Random Forest")
  # pred_store[[model]][[i]] collects predictions for observation i across splits
  pred_store <- lapply(model_names, function(m) vector("list", n))
  names(pred_store) <- model_names

  message("  Empirical bias-variance: running ", n_repeats, " splits...")

  for (r in seq_len(n_repeats)) {
    seed_r <- base_seed + r
    sp <- split_train_test(dat, response = response, seed = seed_r,
                           train_prop = train_prop)
    tr <- sp$train
    te <- sp$test
    test_idx <- as.integer(rownames(te))  # original row indices

    y_test <- te[[response]]
    x_test_mat <- as.matrix(te[, predictors, drop = FALSE])

    # OLS
    fml <- stats::as.formula(paste(response, "~ ."))
    ols_fit <- lm(fml, data = tr)
    ols_pred <- predict(ols_fit, newdata = te)

    # Ridge (alpha=0)
    x_tr <- as.matrix(tr[, predictors, drop = FALSE])
    y_tr <- tr[[response]]
    set.seed(seed_r + 10000)
    ridge_cv <- glmnet::cv.glmnet(x_tr, y_tr, alpha = 0, nfolds = 5,
                                   standardize = TRUE, type.measure = "mse")
    ridge_pred <- as.numeric(predict(ridge_cv, newx = x_test_mat, s = "lambda.min"))

    # Lasso (alpha=1)
    set.seed(seed_r + 20000)
    lasso_cv <- glmnet::cv.glmnet(x_tr, y_tr, alpha = 1, nfolds = 5,
                                   standardize = TRUE, type.measure = "mse")
    lasso_pred <- as.numeric(predict(lasso_cv, newx = x_test_mat, s = "lambda.min"))

    # Decision tree (pruned via 1-SE rule)
    set.seed(seed_r + 40000)
    tree_full <- rpart::rpart(fml, data = tr, method = "anova",
                               control = rpart::rpart.control(
                                 cp = 1e-4, minsplit = 20,
                                 maxdepth = 10, xval = 10))
    cp_tbl <- tree_full$cptable
    best_row <- which.min(cp_tbl[, "xerror"])
    thresh <- cp_tbl[best_row, "xerror"] + cp_tbl[best_row, "xstd"]
    eligible <- which(cp_tbl[, "xerror"] <= thresh)
    cp_1se <- cp_tbl[min(eligible), "CP"]
    tree_pruned <- rpart::prune(tree_full, cp = cp_1se)
    tree_pred <- as.numeric(predict(tree_pruned, newdata = te))

    # Random Forest
    rf_mtry <- max(1, min(RF_DEFAULTS$mtry, p))
    rf_fit <- ranger::ranger(
      fml, data = tr, num.trees = num_trees,
      mtry = rf_mtry, min.node.size = RF_DEFAULTS$min_node_size,
      seed = seed_r, write.forest = TRUE
    )
    rf_pred <- predict(rf_fit, data = te)$predictions

    # Store predictions indexed by original row number
    preds <- list(OLS = ols_pred, Ridge = ridge_pred, Lasso = lasso_pred,
                  `Decision Tree` = tree_pred, `Random Forest` = rf_pred)

    for (m in model_names) {
      for (j in seq_along(test_idx)) {
        idx <- test_idx[j]
        pred_store[[m]][[idx]] <- c(pred_store[[m]][[idx]], preds[[m]][j])
      }
    }

    if (r %% 10 == 0) message("    Split ", r, "/", n_repeats, " done")
  }

  # Compute per-observation bias^2 and variance
  bv_results <- data.frame(Model = character(), Bias2 = numeric(),
                           Variance = numeric(), MSE = numeric(),
                           stringsAsFactors = FALSE)

  for (m in model_names) {
    bias2_vec <- numeric(n)
    var_vec   <- numeric(n)
    count_vec <- integer(n)

    for (i in seq_len(n)) {
      preds_i <- pred_store[[m]][[i]]
      if (length(preds_i) < 2) next
      mean_pred <- mean(preds_i)
      bias2_vec[i] <- (mean_pred - y[i])^2
      var_vec[i]   <- var(preds_i)
      count_vec[i] <- length(preds_i)
    }

    # Only use observations that appeared in test set at least twice
    valid <- count_vec >= 2
    avg_bias2 <- mean(bias2_vec[valid])
    avg_var   <- mean(var_vec[valid])
    avg_mse   <- avg_bias2 + avg_var  # approximate (ignores irreducible)

    bv_results <- rbind(bv_results, data.frame(
      Model = m, Bias2 = avg_bias2, Variance = avg_var, MSE = avg_mse,
      stringsAsFactors = FALSE
    ))
  }

  # Estimate irreducible error from RF residuals (best model, least bias)
  # sigma^2 ~ average variance of y around RF mean prediction
  rf_resid_var <- numeric(n)
  valid_rf <- integer(n)
  for (i in seq_len(n)) {
    preds_i <- pred_store[["Random Forest"]][[i]]
    if (length(preds_i) < 2) next
    rf_resid_var[i] <- mean((preds_i - y[i])^2) - var(preds_i)
    valid_rf[i] <- 1L
  }
  sigma2_est <- max(0, mean(rf_resid_var[valid_rf == 1]))
  bv_results$Irreducible <- sigma2_est

  readr::write_csv(bv_results, file.path(output_dir, "empirical_biasvar.csv"))

  # Stacked bar chart
  fig_path <- file.path(fig_dir, "empirical_biasvar_decomposition.pdf")
  grDevices::pdf(fig_path, width = 12, height = 8, pointsize = 10)
  op <- par(mar = c(8, 5, 4, 2))

  bar_data <- rbind(
    bv_results$Irreducible,
    bv_results$Bias2,
    bv_results$Variance
  )
  colnames(bar_data) <- bv_results$Model

  cols <- c("grey70", "firebrick", "steelblue")
  bp <- barplot(bar_data, beside = FALSE, col = cols, las = 2,
                ylab = "Mean Squared Error",
                main = paste0("Empirical Bias-Variance Decomposition (",
                              n_repeats, " splits)"),
                border = NA)
  legend("topright",
         legend = expression(hat(sigma)^2 ~ "(irreducible)",
                             Bias^2, Variance),
         fill = cols, bty = "n", border = NA)

  par(op)
  grDevices::dev.off()

  message("  Saved empirical_biasvar_decomposition.pdf")

  list(
    results = bv_results,
    sigma2_est = sigma2_est,
    figures = c(empirical_biasvar = fig_path)
  )
}
