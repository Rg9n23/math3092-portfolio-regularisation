# Plot utilities for single-split outputs.

save_all_plots <- function(
    train,
    test,
    results,
    repeated_results = NULL,  # kept for API compatibility, but not plotted here
    response = RESPONSE,
    response_label = NULL,
    fig_dir = FIG_DIR
) {
  assert_response(train, response)
  assert_response(test, response)
  
  if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)
  if (is.null(response_label)) {
    response_label <- gsub("_", " ", response)
  }

  plot_files <- character(0)
  
  pred_df <- data.frame(
    actual = test[[response]],
    OLS = results$linear$pred,
    Ridge = results$regularized$pred$Ridge,
    Lasso = results$regularized$pred$Lasso,
    Elastic_Net = results$regularized$pred$`Elastic Net`,
    Tree = results$tree$pred,
    Random_Forest = results$rf$pred,
    check.names = FALSE
  )
  
  # 1) Predicted vs actual panels
  models <- setdiff(names(pred_df), "actual")
  k <- length(models)
  ncol <- 3
  nrow <- ceiling(k / ncol)
  
  path_scatter <- file.path(fig_dir, "pred_vs_actual_all_models.pdf")
  grDevices::pdf(path_scatter, width = 10, height = (1200 + 350*(nrow-1)) / 220, pointsize = 10)
  op <- par(mfrow = c(nrow, ncol), mar = c(4, 4, 3, 1))
  for (m in models) {
    plot(
      pred_df$actual,
      pred_df[[m]],
      pch = 16,
      cex = 0.6,
      xlab = paste("Actual", response_label),
      ylab = paste("Predicted", response_label),
      main = m
    )
    abline(0, 1, lty = 2, lwd = 1.5)
  }
  par(op)
  grDevices::dev.off()
  plot_files <- c(plot_files, path_scatter)
  
  # 2) RMSE bar plot (best at top)
  path_rmse <- file.path(fig_dir, "model_rmse_bar.pdf")
  grDevices::pdf(path_rmse, width = 9, height = 5.5, pointsize = 10)
  ord <- order(results$model_table$RMSE, decreasing = FALSE)
  op <- par(mar = c(5, 14, 4, 2))
  barplot(
    height = results$model_table$RMSE[ord],
    names.arg = results$model_table$Model[ord],
    horiz = TRUE,
    las = 1,
    xlab = "RMSE",
    main = "Model comparison by RMSE (lower is better)"
  )
  par(op)
  grDevices::dev.off()
  plot_files <- c(plot_files, path_rmse)
  
  # 3) Residuals vs fitted for OLS (simple diagnostic; helps the report feel like research)
  if (!is.null(results$linear$extras$fitted) && !is.null(results$linear$extras$residuals)) {
    path_ols_diag <- file.path(fig_dir, "ols_residuals_vs_fitted.pdf")
    grDevices::pdf(path_ols_diag, width = 8, height = 5.5, pointsize = 10)
    plot(
      results$linear$extras$fitted,
      results$linear$extras$residuals,
      pch = 16,
      cex = 0.6,
      xlab = "Fitted values",
      ylab = "Residuals",
      main = "OLS diagnostic: residuals vs fitted"
    )
    abline(h = 0, lty = 2)
    grDevices::dev.off()
    plot_files <- c(plot_files, path_ols_diag)
  }
  
  list(
    model = NULL,
    pred = NULL,
    metrics = NULL,
    extras = list(files = plot_files)
  )
}