# Polynomial complexity curve: train/test RMSE vs polynomial degree.
# Demonstrates the bias-variance tradeoff on real data.

fit_polynomial_complexity <- function(
    train,
    test,
    response,
    predictor = NULL,
    max_degree = POLY_MAX_DEGREE,
    fig_dir = FIG_DIR,
    output_dir = OUTPUT_DIR
) {
  assert_response(train, response)
  assert_response(test, response)
  if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  y_train <- train[[response]]
  y_test  <- test[[response]]

  # If no predictor specified, pick the one with highest absolute correlation
  predictors <- setdiff(names(train), response)
  if (is.null(predictor)) {
    cors <- vapply(predictors, function(p) abs(cor(train[[p]], y_train)), numeric(1))
    predictor <- names(which.max(cors))
    message("  Polynomial curve: auto-selected predictor '", predictor, "' (|cor| = ",
            round(max(cors), 3), ")")
  }

  x_train <- train[[predictor]]
  x_test  <- test[[predictor]]

  # Cap degree at number of unique training values minus 1
  n_unique <- length(unique(x_train))
  if (max_degree >= n_unique) {
    max_degree <- n_unique - 1
    message("  Capping polynomial degree at ", max_degree, " (", n_unique, " unique values)")
  }

  degrees <- seq_len(max_degree)
  train_rmse <- numeric(max_degree)
  test_rmse  <- numeric(max_degree)

  for (d in degrees) {
    ok <- tryCatch({
      fit <- lm(y_train ~ poly(x_train, degree = d, raw = FALSE))
      pred_train <- predict(fit)
      pred_test  <- predict(fit, newdata = data.frame(x_train = x_test))
      train_rmse[d] <- rmse(y_train, pred_train)
      test_rmse[d]  <- rmse(y_test, pred_test)
      TRUE
    }, error = function(e) {
      message("  poly() failed at degree ", d, ": ", conditionMessage(e))
      FALSE
    })
    if (!ok) {
      # Truncate to degrees that worked
      max_degree <- d - 1L
      degrees <- seq_len(max_degree)
      train_rmse <- train_rmse[degrees]
      test_rmse  <- test_rmse[degrees]
      break
    }
  }

  results <- data.frame(
    degree     = degrees,
    train_rmse = train_rmse,
    test_rmse  = test_rmse
  )

  readr::write_csv(results, file.path(output_dir, "polynomial_complexity.csv"))

  # Plot
  fig_path <- file.path(fig_dir, "polynomial_complexity_curve.pdf")
  grDevices::pdf(fig_path, width = 10, height = 7, pointsize = 10)
  op <- par(mar = c(5, 5, 4, 2))

  ylim <- range(c(train_rmse, test_rmse))
  plot(degrees, test_rmse, type = "b", pch = 16, col = "firebrick",
       xlab = "Polynomial degree", ylab = "RMSE",
       ylim = ylim, main = paste0("Bias-Variance Tradeoff: Polynomial in '",
                                   predictor, "' (", response, ")"),
       lwd = 2)
  lines(degrees, train_rmse, type = "b", pch = 17, col = "steelblue", lwd = 2)

  best_deg <- which.min(test_rmse)
  abline(v = best_deg, lty = 2, col = "grey40")
  text(best_deg, max(ylim) * 0.95, paste("Best degree =", best_deg),
       pos = 4, cex = 0.8, col = "grey40")

  legend("topright", legend = c("Test RMSE", "Training RMSE"),
         col = c("firebrick", "steelblue"), pch = c(16, 17), lwd = 2, bty = "n")

  par(op)
  grDevices::dev.off()

  message("  Saved polynomial_complexity_curve.pdf")

  list(
    results = results,
    predictor = predictor,
    best_degree = best_deg,
    figures = c(polynomial_complexity = fig_path)
  )
}
