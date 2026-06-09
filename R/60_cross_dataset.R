# Cross-dataset synthesis: compare model performance and eigenvalue spectra
# across all datasets in the pipeline.

run_cross_dataset_synthesis <- function(
    output_dir = OUTPUT_DIR,
    fig_dir = FIG_DIR,
    datasets = DATASETS
) {
  cross_out <- file.path(output_dir, "cross_dataset")
  cross_fig <- file.path(fig_dir, "cross_dataset")
  if (!dir.exists(cross_out)) dir.create(cross_out, recursive = TRUE)
  if (!dir.exists(cross_fig)) dir.create(cross_fig, recursive = TRUE)

  ds_names <- names(datasets)

  # --- 1. Read all model_table.csv files ---
  model_tables <- list()
  for (nm in ds_names) {
    path <- file.path(output_dir, nm, "model_table.csv")
    if (!file.exists(path)) {
      message("  Skipping ", nm, ": model_table.csv not found")
      next
    }
    tbl <- readr::read_csv(path, show_col_types = FALSE)
    tbl$dataset <- nm
    tbl$label <- datasets[[nm]]$label
    model_tables[[nm]] <- tbl
  }
  if (length(model_tables) == 0) stop("No model_table.csv files found.")
  all_models <- do.call(rbind, model_tables)

  # --- 2. Read all eigenvalue_analysis.csv files ---
  eigen_tables <- list()
  for (nm in ds_names) {
    path <- file.path(output_dir, nm, "eigenvalue_analysis.csv")
    if (!file.exists(path)) next
    tbl <- readr::read_csv(path, show_col_types = FALSE)
    tbl$dataset <- nm
    tbl$label <- datasets[[nm]]$label
    eigen_tables[[nm]] <- tbl
  }
  all_eigen <- if (length(eigen_tables) > 0) do.call(rbind, eigen_tables) else NULL

  # --- 3. Master RMSE matrix: dataset x model ---
  # Simplify model names (strip hyperparameter details)
  simplify_model <- function(m) {
    m <- sub("\\s*\\(.*", "", m)
    m
  }

  # For the master table, use the best variant of each model family
  # (i.e., lowest RMSE among Ridge lambda.min/1se, Lasso lambda.min/1se, etc.)
  family_map <- function(m) {
    if (grepl("^OLS", m)) return("OLS")
    if (grepl("^Ridge", m)) return("Ridge")
    if (grepl("^Lasso", m)) return("Lasso")
    if (grepl("^Elastic", m)) return("Elastic Net")
    if (grepl("^Decision", m)) return("Decision Tree")
    if (grepl("^Random", m)) return("Random Forest")
    return(m)
  }

  all_models$family <- vapply(all_models$model, family_map, character(1))

  # Best RMSE per family per dataset
  best_per_family <- do.call(rbind, lapply(split(all_models, list(all_models$dataset, all_models$family)), function(d) {
    if (nrow(d) == 0) return(NULL)
    d[which.min(d$test_rmse), , drop = FALSE]
  }))

  # Pivot to wide: dataset x model family
  families <- c("OLS", "Ridge", "Lasso", "Elastic Net", "Decision Tree", "Random Forest")
  rmse_matrix <- data.frame(dataset = character(), label = character(), stringsAsFactors = FALSE)
  for (nm in ds_names) {
    sub <- best_per_family[best_per_family$dataset == nm, ]
    if (nrow(sub) == 0) next
    row <- data.frame(dataset = nm, label = datasets[[nm]]$label, stringsAsFactors = FALSE)
    for (fam in families) {
      val <- sub$test_rmse[sub$family == fam]
      row[[fam]] <- if (length(val) == 1) val else NA_real_
    }
    rmse_matrix <- rbind(rmse_matrix, row)
  }

  readr::write_csv(rmse_matrix, file.path(cross_out, "master_rmse_matrix.csv"))
  message("  Saved master_rmse_matrix.csv")

  # --- 4. Summary: winner per dataset + kappa ---
  summary_rows <- list()
  for (i in seq_len(nrow(rmse_matrix))) {
    nm <- rmse_matrix$dataset[i]
    vals <- unlist(rmse_matrix[i, families])
    winner <- names(which.min(vals))
    best_rmse <- min(vals, na.rm = TRUE)
    ols_rmse <- rmse_matrix[i, "OLS"]

    kappa_val <- NA_real_
    if (!is.null(all_eigen)) {
      eig_sub <- all_eigen[all_eigen$dataset == nm, ]
      if (nrow(eig_sub) > 0) {
        svs <- eig_sub$singular_value
        kappa_val <- max(svs) / min(svs)
      }
    }

    # Best regularised RMSE (Ridge, Lasso, or Elastic Net)
    reg_rmses <- unlist(rmse_matrix[i, c("Ridge", "Lasso", "Elastic Net")])
    best_reg_rmse <- min(reg_rmses, na.rm = TRUE)
    reg_improvement <- ols_rmse - best_reg_rmse

    summary_rows[[i]] <- data.frame(
      dataset = nm,
      label = rmse_matrix$label[i],
      n = NA_integer_,
      p = NA_integer_,
      kappa = kappa_val,
      log_kappa = log(kappa_val),
      winner = winner,
      best_rmse = best_rmse,
      ols_rmse = ols_rmse,
      best_reg_rmse = best_reg_rmse,
      reg_improvement = reg_improvement,
      stringsAsFactors = FALSE
    )
  }
  summary_df <- do.call(rbind, summary_rows)

  # Fill in n and p from the actual data
  for (i in seq_len(nrow(summary_df))) {
    nm <- summary_df$dataset[i]
    ds <- datasets[[nm]]
    if (file.exists(ds$file)) {
      dat <- readr::read_csv(ds$file, show_col_types = FALSE)
      names(dat) <- clean_names(names(dat))
      if (is.function(ds$preprocess)) dat <- ds$preprocess(dat)
      dat <- dat[complete.cases(dat), , drop = FALSE]
      summary_df$n[i] <- nrow(dat)
      summary_df$p[i] <- ncol(dat) - 1L
    }
  }

  readr::write_csv(summary_df, file.path(cross_out, "dataset_summary.csv"))
  message("  Saved dataset_summary.csv")

  # --- 5. Scatter plot: regularisation improvement vs log(kappa) ---
  if (sum(is.finite(summary_df$log_kappa) & is.finite(summary_df$reg_improvement)) >= 2) {
    scatter_path <- file.path(cross_fig, "reg_improvement_vs_kappa.pdf")
    grDevices::pdf(scatter_path, width = 10, height = 7, pointsize = 10)
    op <- par(mar = c(5, 5, 4, 8), xpd = TRUE)
    plot(
      summary_df$log_kappa,
      summary_df$reg_improvement,
      pch = 16, cex = 1.5,
      xlab = expression(log(kappa(X))),
      ylab = "Regularisation improvement (OLS RMSE - best reg. RMSE)",
      main = "Regularisation benefit vs design matrix conditioning"
    )
    text(
      summary_df$log_kappa,
      summary_df$reg_improvement,
      labels = summary_df$label,
      pos = 4, cex = 0.5, offset = 0.4
    )
    abline(h = 0, lty = 2, col = "grey60")
    par(op)
    grDevices::dev.off()
    message("  Saved reg_improvement_vs_kappa.pdf")
  }

  # --- 6. Win-rate table from repeated splits ---
  win_rate_tables <- list()
  for (nm in ds_names) {
    path <- file.path(output_dir, nm, "repeated_splits_results.csv")
    if (!file.exists(path)) next
    res <- readr::read_csv(path, show_col_types = FALSE)
    # Winner per repeat
    winners <- by(res, res$Repeat, function(d) d$Model[which.min(d$RMSE)][1])
    winners <- as.character(winners)
    n_reps <- length(unique(res$Repeat))
    tbl <- as.data.frame(table(winners), stringsAsFactors = FALSE)
    names(tbl) <- c("Model", "wins")
    tbl$win_pct <- tbl$wins / n_reps * 100
    tbl$dataset <- nm
    tbl$n_repeats <- n_reps
    win_rate_tables[[nm]] <- tbl
  }

  if (length(win_rate_tables) > 0) {
    all_wins <- do.call(rbind, win_rate_tables)

    # Pivot: model x dataset win percentages
    all_fams <- sort(unique(all_wins$Model))
    win_matrix <- data.frame(Model = all_fams, stringsAsFactors = FALSE)
    for (nm in ds_names) {
      sub <- all_wins[all_wins$dataset == nm, ]
      col <- rep(0, length(all_fams))
      names(col) <- all_fams
      for (j in seq_len(nrow(sub))) {
        col[sub$Model[j]] <- sub$win_pct[j]
      }
      win_matrix[[datasets[[nm]]$label]] <- col
    }
    # Add overall average
    numeric_cols <- setdiff(names(win_matrix), "Model")
    win_matrix$Average <- rowMeans(win_matrix[, numeric_cols, drop = FALSE])

    readr::write_csv(win_matrix, file.path(cross_out, "win_rate_matrix.csv"))
    message("  Saved win_rate_matrix.csv")
  }

  # --- 7. Eigenvalue spectrum comparison plot ---
  if (!is.null(all_eigen)) {
    spec_path <- file.path(cross_fig, "eigenvalue_spectra.pdf")
    ds_with_eigen <- unique(all_eigen$dataset)
    n_ds <- length(ds_with_eigen)
    cols <- grDevices::rainbow(n_ds, s = 0.7, v = 0.8)

    grDevices::pdf(spec_path, width = 12, height = 7, pointsize = 10)
    op <- par(mar = c(5, 5, 4, 8), xpd = TRUE)

    # Get range for axes
    all_log_sv <- log(all_eigen$singular_value)
    max_comp <- max(all_eigen$component)
    ylim <- range(all_log_sv[is.finite(all_log_sv)])

    plot(NULL, xlim = c(1, max_comp), ylim = ylim,
         xlab = "Component", ylab = "log(singular value)",
         main = "Eigenvalue spectra across datasets")

    for (i in seq_along(ds_with_eigen)) {
      nm <- ds_with_eigen[i]
      sub <- all_eigen[all_eigen$dataset == nm, ]
      lines(sub$component, log(sub$singular_value), col = cols[i], lwd = 2)
      points(sub$component, log(sub$singular_value), col = cols[i], pch = 16, cex = 0.5)
    }

    labels <- vapply(ds_with_eigen, function(nm) datasets[[nm]]$label, character(1))
    legend("topright", inset = c(-0.2, 0), legend = labels,
           col = cols, lwd = 2, pch = 16, cex = 0.7, bty = "n")
    par(op)
    grDevices::dev.off()
    message("  Saved eigenvalue_spectra.pdf")
  }

  message("  Cross-dataset synthesis complete.")

  invisible(list(
    rmse_matrix = rmse_matrix,
    summary = summary_df,
    eigen = all_eigen
  ))
}
