# Data loading and cleaning helpers.

load_and_clean_data <- function(path, response = RESPONSE, preprocess = NULL) {
  if (!file.exists(path)) {
    stop("Data file not found: ", path)
  }
  
  dat <- readr::read_csv(path, show_col_types = FALSE)
  names(dat) <- clean_names(names(dat))
  dat <- as.data.frame(dat)
  
  # Basic integrity checks
  if (any(names(dat) == "")) {
    stop("Empty column name detected after cleaning.")
  }
  if (any(duplicated(names(dat)))) {
    dup <- unique(names(dat)[duplicated(names(dat))])
    stop("Duplicate column name(s) after cleaning: ", paste(dup, collapse = ", "))
  }
  
  # Apply dataset-specific preprocessing if provided
  if (is.function(preprocess)) {
    dat <- preprocess(dat)
  }

  assert_response(dat, response)

  # Enforce numeric response
  if (!is.numeric(dat[[response]])) {
    suppressWarnings(dat[[response]] <- as.numeric(dat[[response]]))
  }
  if (!is.numeric(dat[[response]])) {
    stop("Response column '", response, "' is not numeric after coercion.")
  }
  
  # Ensure predictors are numeric (no factors/characters slipping in)
  pred_names <- setdiff(names(dat), response)
  non_numeric <- pred_names[!vapply(dat[pred_names], is.numeric, logical(1))]
  if (length(non_numeric) > 0) {
    # try coercion
    for (nm in non_numeric) {
      suppressWarnings(dat[[nm]] <- as.numeric(dat[[nm]]))
    }
    still_bad <- pred_names[!vapply(dat[pred_names], is.numeric, logical(1))]
    if (length(still_bad) > 0) {
      stop("Non-numeric predictor(s) after coercion: ", paste(still_bad, collapse = ", "))
    }
  }
  
  # Missing value check
  na_counts <- colSums(is.na(dat))
  if (any(na_counts > 0)) {
    bad <- names(na_counts)[na_counts > 0]
    stop("Missing values found in: ", paste(bad, collapse = ", "),
         ". Decide an imputation/removal strategy explicitly.")
  }
  
  dat
}