# Train/test split helper.

split_train_test <- function(dat,
                             response = RESPONSE,
                             seed = SEED,
                             train_prop = TRAIN_PROP,
                             stratify = FALSE,
                             n_bins = 10) {
  assert_response(dat, response)
  
  if (!(train_prop > 0 && train_prop < 1)) {
    stop("train_prop must be in (0, 1).")
  }
  
  n <- nrow(dat)
  if (n < 20) stop("Dataset too small for reliable train/test split (n=", n, ").")
  
  set.seed(seed)
  n_train <- round(train_prop * n)
  
  if (!stratify) {
    idx_train <- sample.int(n, size = n_train)
  } else {
    y <- dat[[response]]
    if (!is.numeric(y)) stop("Cannot stratify: response is not numeric.")
    
    # Bin y into quantiles; ensure unique breaks
    probs <- seq(0, 1, length.out = n_bins + 1)
    brks <- unique(as.numeric(stats::quantile(y, probs = probs, na.rm = TRUE, type = 7)))
    if (length(brks) < 3) {
      # fallback if y has too many ties
      idx_train <- sample.int(n, size = n_train)
    } else {
      bins <- cut(y, breaks = brks, include.lowest = TRUE, labels = FALSE)
      idx_train <- integer(0)
      
      # sample proportionally within bins
      for (b in sort(unique(bins))) {
        idx_b <- which(bins == b)
        k <- round(length(idx_b) * train_prop)
        if (k > 0) idx_train <- c(idx_train, sample(idx_b, size = min(k, length(idx_b))))
      }
      
      # adjust size to match n_train exactly
      idx_train <- unique(idx_train)
      if (length(idx_train) > n_train) {
        idx_train <- sample(idx_train, size = n_train)
      } else if (length(idx_train) < n_train) {
        remaining <- setdiff(seq_len(n), idx_train)
        idx_train <- c(idx_train, sample(remaining, size = n_train - length(idx_train)))
      }
    }
  }
  
  idx_train <- sort(idx_train)
  idx_test <- setdiff(seq_len(n), idx_train)
  
  train <- dat[idx_train, , drop = FALSE]
  test  <- dat[idx_test,  , drop = FALSE]
  
  list(
    train = train,
    test = test,
    idx_train = idx_train,
    idx_test = idx_test,
    seed = seed,
    train_prop = train_prop,
    stratify = stratify,
    n_bins = n_bins
  )
}