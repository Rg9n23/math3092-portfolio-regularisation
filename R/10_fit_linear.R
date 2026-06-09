# Ordinary least squares model.

fit_linear_model <- function(train, test, response = RESPONSE) {
  assert_response(train, response)
  assert_response(test, response)
  
  fml <- stats::as.formula(paste(response, "~ ."))
  model <- stats::lm(fml, data = train)
  
  pred_test <- as.numeric(stats::predict(model, newdata = test))
  if (any(!is.finite(pred_test))) {
    stop("OLS prediction produced non-finite values on test set.")
  }
  test_metrics <- calc_metrics(test[[response]], pred_test)
  
  # Training metrics (useful for diagnostics and comparison)
  pred_train <- as.numeric(stats::predict(model, newdata = train))
  if (any(!is.finite(pred_train))) {
    stop("OLS prediction produced non-finite values on training set.")
  }
  train_metrics <- calc_metrics(train[[response]], pred_train)
  
  list(
    model = model,
    pred = pred_test,
    metrics = test_metrics,
    extras = list(
      formula = fml,
      coefficients = stats::coef(model),
      train_metrics = train_metrics,
      fitted = stats::fitted(model),
      residuals = stats::residuals(model)
    )
  )
}