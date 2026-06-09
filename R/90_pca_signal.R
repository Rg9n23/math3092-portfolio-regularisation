# PCA signal projection: how much response signal aligns with each principal component.

compute_pca_signal <- function(
    train,
    test,
    response,
    fig_dir = FIG_DIR,
    output_dir = OUTPUT_DIR
) {
  assert_response(train, response)
  if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  predictors <- setdiff(names(train), response)
  x_train <- as.matrix(train[, predictors, drop = FALSE])
  y_train <- train[[response]]
  p <- ncol(x_train)

  # Centre and scale
  x_scaled <- scale(x_train)
  col_means <- attr(x_scaled, "scaled:center")
  col_sds   <- attr(x_scaled, "scaled:scale")

  # SVD
  sv <- svd(x_scaled)
  # PC scores: Z = X_scaled %*% V
  z_train <- x_scaled %*% sv$v

  # Regress y on each PC individually
  ss_tot <- sum((y_train - mean(y_train))^2)
  r2_per_pc <- numeric(p)
  for (j in seq_len(p)) {
    fit_j <- lm(y_train ~ z_train[, j])
    r2_per_pc[j] <- 1 - sum(residuals(fit_j)^2) / ss_tot
  }

  # Cumulative R^2 from regressing y on PCs 1..k
  cum_r2 <- numeric(p)
  for (k in seq_len(p)) {
    fit_k <- lm(y_train ~ z_train[, 1:k])
    cum_r2[k] <- 1 - sum(residuals(fit_k)^2) / ss_tot
  }

  # Eigenvalues (d^2)
  d_sq <- sv$d^2
  d_sq_norm <- d_sq / sum(d_sq) * 100  # % variance explained

  results <- data.frame(
    pc = seq_len(p),
    eigenvalue = d_sq,
    pct_var_explained = d_sq_norm,
    r2_individual = r2_per_pc,
    r2_cumulative = cum_r2
  )

  readr::write_csv(results, file.path(output_dir, "pca_signal_projection.csv"))

  # Two-panel plot: eigenvalue spectrum + signal alignment
  fig_path <- file.path(fig_dir, "pca_signal_alignment.pdf")
  grDevices::pdf(fig_path, width = 14, height = 7, pointsize = 10)
  op <- par(mfrow = c(1, 2), mar = c(5, 5, 4, 2))

  # Panel 1: Eigenvalue spectrum
  barplot(d_sq_norm, names.arg = seq_len(p), col = "steelblue",
          xlab = "Principal component", ylab = "% variance in X",
          main = "Eigenvalue spectrum (data variance)")

  # Panel 2: Signal alignment
  plot(seq_len(p), r2_per_pc * 100, type = "h", lwd = 3, col = "firebrick",
       xlab = "Principal component", ylab = expression(R^2 ~ "(% of response variance)"),
       main = "Signal alignment with PCs",
       ylim = c(0, max(r2_per_pc * 100) * 1.1))
  points(seq_len(p), r2_per_pc * 100, pch = 16, col = "firebrick", cex = 0.8)

  par(op)
  grDevices::dev.off()
  message("  Saved pca_signal_alignment.pdf")

  list(
    results = results,
    figures = c(pca_signal = fig_path)
  )
}
