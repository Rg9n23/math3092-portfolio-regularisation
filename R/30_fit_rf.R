# Random forest regression via ranger with interpretability outputs.

fit_rf_model <- function(
    train,
    test,
    response = RESPONSE,
    seed = SEED,
    mtry = RF_DEFAULTS$mtry,
    min_node_size = RF_DEFAULTS$min_node_size,
    num_trees = RF_DEFAULTS$num_trees,
    fig_dir = FIG_DIR,
    output_dir = OUTPUT_DIR,
    pdp_features = c("age", "water", "cement"),
    pdp_grid_resolution = 40,
    pdp_2d = list(c("age", "water")),
    save_artifacts = TRUE
) {
  assert_response(train, response)
  assert_response(test, response)
  
  if (save_artifacts) {
    if (!requireNamespace("pdp", quietly = TRUE)) {
      stop("Package 'pdp' is required for RF partial dependence outputs.")
    }
    if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)
    if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  }
  
  max_mtry <- ncol(train) - 1
  mtry <- max(1, min(mtry, max_mtry))
  
  fml <- stats::as.formula(paste(response, "~ ."))
  model <- ranger::ranger(
    formula = fml,
    data = train,
    num.trees = num_trees,
    mtry = mtry,
    min.node.size = min_node_size,
    importance = "permutation",
    seed = seed
  )
  
  pred_test <- as.numeric(stats::predict(model, data = test)$predictions)
  test_metrics <- calc_metrics(test[[response]], pred_test)
  
  pred_train <- as.numeric(stats::predict(model, data = train)$predictions)
  train_metrics <- calc_metrics(train[[response]], pred_train)
  
  rf_summary <- data.frame(
    Model = "Random Forest",
    RMSE_test = unname(test_metrics["RMSE"]),
    MAE_test  = unname(test_metrics["MAE"]),
    R2_test   = unname(test_metrics["R2"]),
    RMSE_train = unname(train_metrics["RMSE"]),
    MAE_train  = unname(train_metrics["MAE"]),
    R2_train   = unname(train_metrics["R2"]),
    mtry = mtry,
    min_node_size = min_node_size,
    num_trees = num_trees,
    importance = "permutation",
    seed = seed,
    row.names = NULL
  )
  
  vi <- model$variable.importance
  vi <- vi[order(vi, decreasing = TRUE)]
  
  artifact_paths <- list()
  pdp_results <- list()
  
  ranger_pred_fun <- function(object, newdata) {
    as.numeric(stats::predict(object, data = newdata)$predictions)
  }
  
  if (save_artifacts) {
    rf_summary_path <- file.path(output_dir, "rf_summary.csv")
    readr::write_csv(rf_summary, rf_summary_path)
    
    # Permutation variable importance barplot.
    vi_path <- file.path(fig_dir, "rf_var_importance.pdf")
    grDevices::pdf(vi_path, width = 10, height = 7, pointsize = 10)
    op <- par(mar = c(5, 14, 4, 2))
    barplot(
      height = vi,
      names.arg = names(vi),
      horiz = TRUE,
      las = 1,
      xlab = "Permutation importance",
      main = "Random forest permutation importance"
    )
    par(op)
    grDevices::dev.off()
    
    artifact_paths$rf_summary <- rf_summary_path
    artifact_paths$rf_var_importance <- vi_path
    
    missing_features <- setdiff(pdp_features, names(train))
    if (length(missing_features) > 0) {
      stop("Missing required PDP feature(s): ", paste(missing_features, collapse = ", "), ".")
    }
    
    # 1D PDPs
    for (feature in pdp_features) {
      pd <- pdp::partial(
        object = model,
        pred.var = feature,
        train = train,
        pred.fun = ranger_pred_fun,
        grid.resolution = pdp_grid_resolution
      )
      
      pd <- as.data.frame(pd)
      ord <- order(pd[[feature]])
      x <- pd[[feature]][ord]
      y <- pd$yhat[ord]
      
      pdp_path <- file.path(fig_dir, paste0("pdp_", feature, "_rf.pdf"))
      grDevices::pdf(pdp_path, width = 10, height = 7, pointsize = 10)
      plot(
        x, y,
        type = "l",
        lwd = 2,
        xlab = feature,
        ylab = paste("Partial dependence (", response, ")"),
        main = paste("Random forest PDP:", feature)
      )
      points(x, y, pch = 16, cex = 0.5)
      grDevices::dev.off()
      
      pdp_results[[feature]] <- pd
      artifact_paths[[paste0("pdp_", feature)]] <- pdp_path
    }
    
    # 2D PDP(s) for interaction evidence
    for (pair in pdp_2d) {
      if (length(pair) != 2) next
      if (!all(pair %in% names(train))) next
      
      pd2 <- pdp::partial(
        object = model,
        pred.var = pair,
        train = train,
        pred.fun = ranger_pred_fun,
        grid.resolution = min(25, pdp_grid_resolution) # keep size reasonable
      )
      
      pd2 <- as.data.frame(pd2)
      f1 <- pair[1]; f2 <- pair[2]
      
      zmat <- with(pd2, tapply(yhat, list(get(f1), get(f2)), mean))
      xg <- sort(unique(pd2[[f1]]))
      yg <- sort(unique(pd2[[f2]]))
      
      pdp2_path <- file.path(fig_dir, paste0("pdp2_", f1, "_", f2, "_rf.pdf"))
      grDevices::pdf(pdp2_path, width = 10, height = 7, pointsize = 10)
      image(xg, yg, zmat,
            xlab = f1, ylab = f2,
            main = paste("Random forest 2D PDP:", f1, "x", f2))
      contour(xg, yg, zmat, add = TRUE)
      grDevices::dev.off()
      
      pdp_results[[paste0(f1, "_", f2)]] <- pd2
      artifact_paths[[paste0("pdp2_", f1, "_", f2)]] <- pdp2_path
    }
  }
  
  list(
    model = model,
    pred = pred_test,
    metrics = test_metrics,
    extras = list(
      variable_importance = vi,
      rf_summary = rf_summary,
      pdp = pdp_results,
      artifacts = artifact_paths,
      train_metrics = train_metrics,
      hyperparameters = list(
        mtry = mtry,
        min_node_size = min_node_size,
        num_trees = num_trees,
        importance = "permutation",
        seed = seed
      )
    )
  )
}