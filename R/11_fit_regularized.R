# Regularized linear models via glmnet (ridge, lasso, elastic net).

fit_regularized_models <- function(
    train,
    test,
    response = RESPONSE,
    seed = SEED,
    nfolds = N_FOLDS_GLMNET,
    alpha_grid = seq(0.1, 0.9, by = 0.1),
    fig_dir = FIG_DIR,
    output_dir = OUTPUT_DIR,
    save_artifacts = TRUE
) {
  assert_response(train, response)
  assert_response(test, response)
  
  if (length(alpha_grid) < 1) stop("alpha_grid must contain at least one value.")
  
  if (save_artifacts) {
    if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)
    if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  }
  
  x_train <- as.matrix(train[, setdiff(names(train), response), drop = FALSE])
  y_train <- train[[response]]
  x_test  <- as.matrix(test[,  setdiff(names(test),  response), drop = FALSE])
  y_test  <- test[[response]]
  
  fit_cv <- function(alpha_value) {
    set.seed(seed)
    glmnet::cv.glmnet(
      x = x_train,
      y = y_train,
      alpha = alpha_value,
      nfolds = nfolds,
      standardize = TRUE,
      type.measure = "mse"
    )
  }
  
  evaluate <- function(cv_fit, x, y, lambda_rule = "lambda.min") {
    pred <- as.numeric(stats::predict(cv_fit, newx = x, s = lambda_rule))
    list(pred = pred, metrics = calc_metrics(y, pred))
  }
  
  # Ridge (alpha = 0)
  cv_ridge <- fit_cv(alpha_value = 0)
  ridge_test <- evaluate(cv_ridge, x_test, y_test, "lambda.min")
  ridge_train <- evaluate(cv_ridge, x_train, y_train, "lambda.min")
  
  # Lasso (alpha = 1)
  cv_lasso <- fit_cv(alpha_value = 1)
  lasso_test <- evaluate(cv_lasso, x_test, y_test, "lambda.min")
  lasso_train <- evaluate(cv_lasso, x_train, y_train, "lambda.min")
  
  # Elastic net: choose alpha by CV error at lambda.min (training CV only)
  enet_cv_list <- lapply(alpha_grid, fit_cv)
  
  enet_grid <- do.call(
    rbind,
    lapply(seq_along(alpha_grid), function(i) {
      cv_fit <- enet_cv_list[[i]]
      
      # cv error corresponding to lambda.min
      idx_min <- which(cv_fit$lambda == cv_fit$lambda.min)[1]
      cvm_at_lam_min <- cv_fit$cvm[idx_min]
      
      data.frame(
        alpha = alpha_grid[i],
        lambda_min = cv_fit$lambda.min,
        lambda_1se = cv_fit$lambda.1se,
        cvm_lambda_min = cvm_at_lam_min,
        cv_rmse_lambda_min = sqrt(cvm_at_lam_min),
        row.names = NULL
      )
    })
  )
  
  best_enet_idx <- which.min(enet_grid$cvm_lambda_min)
  best_enet_alpha <- enet_grid$alpha[best_enet_idx]
  cv_enet <- enet_cv_list[[best_enet_idx]]
  
  enet_test <- evaluate(cv_enet, x_test, y_test, "lambda.min")
  enet_train <- evaluate(cv_enet, x_train, y_train, "lambda.min")
  
  pred <- data.frame(
    Ridge = ridge_test$pred,
    Lasso = lasso_test$pred,
    `Elastic Net` = enet_test$pred,
    check.names = FALSE
  )
  
  metrics <- rbind(
    as_metric_row("Ridge", ridge_test$metrics),
    as_metric_row("Lasso", lasso_test$metrics),
    as_metric_row("Elastic Net", enet_test$metrics)
  )
  
  lambdas <- data.frame(
    Model = c("Ridge", "Lasso", "Elastic Net"),
    alpha = c(0, 1, best_enet_alpha),
    lambda_min = c(cv_ridge$lambda.min, cv_lasso$lambda.min, cv_enet$lambda.min),
    lambda_1se = c(cv_ridge$lambda.1se, cv_lasso$lambda.1se, cv_enet$lambda.1se),
    row.names = NULL
  )
  
  # Lasso coefficient table at lambda.min and lambda.1se
  coef_min <- as.matrix(stats::coef(cv_lasso, s = "lambda.min"))
  coef_1se <- as.matrix(stats::coef(cv_lasso, s = "lambda.1se"))
  
  lasso_coef_table <- rbind(
    data.frame(
      term = rownames(coef_min),
      lambda_rule = "lambda.min",
      lambda = cv_lasso$lambda.min,
      coefficient = as.numeric(coef_min[, 1]),
      row.names = NULL
    ),
    data.frame(
      term = rownames(coef_1se),
      lambda_rule = "lambda.1se",
      lambda = cv_lasso$lambda.1se,
      coefficient = as.numeric(coef_1se[, 1]),
      row.names = NULL
    )
  )
  
  artifact_paths <- list()
  
  if (save_artifacts) {
    summary_path <- file.path(output_dir, "regularized_summary.csv")
    coef_path <- file.path(output_dir, "lasso_coefficients.csv")
    lasso_cv_fig <- file.path(fig_dir, "lasso_cv_rmse.pdf")
    lasso_path_fig <- file.path(fig_dir, "lasso_coef_path.pdf")
    
    readr::write_csv(metrics, summary_path)
    readr::write_csv(lasso_coef_table, coef_path)
    
    # Lasso CV curve in RMSE units.
    cv_rmse <- sqrt(cv_lasso$cvm)
    cv_low  <- sqrt(cv_lasso$cvlo)
    cv_high <- sqrt(cv_lasso$cvup)
    log_lambda <- log(cv_lasso$lambda)
    
    grDevices::pdf(lasso_cv_fig, width = 10, height = 7, pointsize = 10)
    plot(
      log_lambda, cv_rmse,
      type = "b", pch = 16,
      xlab = "log(lambda)",
      ylab = "CV RMSE",
      main = "Lasso: 10-fold CV RMSE vs log(lambda)"
    )
    segments(log_lambda, cv_low, log_lambda, cv_high)
    graphics::abline(v = log(cv_lasso$lambda.min), lty = 2, lwd = 2)
    graphics::abline(v = log(cv_lasso$lambda.1se), lty = 2, lwd = 2)
    legend(
      "topright",
      legend = c(
        sprintf("lambda.min = %.4g", cv_lasso$lambda.min),
        sprintf("lambda.1se = %.4g", cv_lasso$lambda.1se)
      ),
      lty = 2, lwd = 2, bty = "n"
    )
    grDevices::dev.off()
    
    # Lasso coefficient path.
    lasso_path_fit <- glmnet::glmnet(
      x = x_train,
      y = y_train,
      alpha = 1,
      standardize = TRUE
    )
    
    grDevices::pdf(lasso_path_fig, width = 10, height = 7, pointsize = 10)
    plot(lasso_path_fit, xvar = "lambda", label = FALSE, main = "Lasso coefficient path")
    graphics::abline(v = log(cv_lasso$lambda.min), lty = 2, lwd = 2)
    graphics::abline(v = log(cv_lasso$lambda.1se), lty = 2, lwd = 2)
    legend("topright", legend = c("lambda.min", "lambda.1se"), lty = 2, lwd = 2, bty = "n")
    grDevices::dev.off()
    
    artifact_paths <- list(
      regularized_summary = summary_path,
      lasso_coefficients = coef_path,
      lasso_cv_rmse = lasso_cv_fig,
      lasso_coef_path = lasso_path_fig
    )
  }
  
  list(
    model = list(
      Ridge = cv_ridge,
      Lasso = cv_lasso,
      `Elastic Net` = cv_enet
    ),
    pred = pred,
    metrics = metrics,
    extras = list(
      lambdas = lambdas,
      enet_grid = enet_grid,
      selected_enet_alpha = best_enet_alpha,
      lasso_coefficients = lasso_coef_table,
      train_metrics = list(
        Ridge = ridge_train$metrics,
        Lasso = lasso_train$metrics,
        `Elastic Net` = enet_train$metrics
      ),
      artifacts = artifact_paths
    )
  )
}