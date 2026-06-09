# Global configuration and shared helpers for the modelling pipeline.

PROJECT_DIR <- normalizePath(".", winslash = "/", mustWork = FALSE)
R_DIR <- file.path(PROJECT_DIR, "R")
FIG_DIR <- file.path(PROJECT_DIR, "figs")
OUTPUT_DIR <- file.path(PROJECT_DIR, "outputs")
DATA_DIR <- file.path(PROJECT_DIR, "data")


SEED <- 3092
TRAIN_PROP <- 0.8
RESPONSE <- "concrete_compressive_strength"

# --- Dataset configurations ---------------------------------------------------
# Each config is a named list used by run_pipeline() in main.R.

DATASETS <- list(
  concrete = list(
    name = "concrete",
    label = "Concrete",
    file = file.path(DATA_DIR, "Concrete_Data.csv"),
    response = "concrete_compressive_strength",
    preprocess = NULL,
    pdp_features = c("age", "water", "cement")
  ),
  communities_crime = list(
    name = "communities_crime",
    label = "Communities & Crime",
    file = file.path(DATA_DIR, "Communities_Crime.csv"),
    response = "violentcrimesperpop",
    preprocess = function(dat) {
      # Drop non-predictive identifiers
      drop_ids <- c("state", "county", "community", "communityname", "fold")
      dat <- dat[, !names(dat) %in% drop_ids, drop = FALSE]

      # Drop columns with >50% missing (LEMAS/police columns, ~84% missing)
      na_frac <- colMeans(is.na(dat))
      high_na <- names(na_frac)[na_frac > 0.5]
      if (length(high_na) > 0) {
        message("  Dropping ", length(high_na), " columns with >50% missing: ",
                paste(head(high_na, 5), collapse = ", "),
                if (length(high_na) > 5) paste0(", ... (", length(high_na), " total)"))
        dat <- dat[, !names(dat) %in% high_na, drop = FALSE]
      }

      # Remove any remaining rows with NA (OtherPerCap has 1 missing)
      n_before <- nrow(dat)
      dat <- dat[complete.cases(dat), , drop = FALSE]
      n_dropped <- n_before - nrow(dat)
      if (n_dropped > 0) {
        message("  Dropped ", n_dropped, " rows with remaining NAs")
      }

      dat
    },
    pdp_features = NULL  # set dynamically after fitting (top 3 by importance)
  ),
  wine_quality = list(
    name = "wine_quality",
    label = "Wine Quality (Red)",
    file = file.path(DATA_DIR, "Wine_Quality_Red.csv"),
    response = "quality",
    preprocess = NULL,
    pdp_features = NULL
  ),
  energy_efficiency = list(
    name = "energy_efficiency",
    label = "Energy Efficiency",
    file = file.path(DATA_DIR, "Energy_Efficiency.csv"),
    response = "y1",
    preprocess = function(dat) {
      # Drop Y2 (cooling load) — we model Y1 (heating load) only
      dat$y2 <- NULL
      dat
    },
    pdp_features = NULL
  ),
  superconductor = list(
    name = "superconductor",
    label = "Superconductor",
    file = file.path(DATA_DIR, "Superconductor.csv"),
    response = "critical_temp",
    preprocess = NULL,
    pdp_features = NULL
  ),
  airfoil = list(
    name = "airfoil",
    label = "Airfoil Self-Noise",
    file = file.path(DATA_DIR, "Airfoil_Self_Noise.csv"),
    response = "sound_pressure_level",
    preprocess = NULL,
    pdp_features = NULL
  ),
  bike_sharing = list(
    name = "bike_sharing",
    label = "Bike Sharing (Hourly)",
    file = file.path(DATA_DIR, "Bike_Sharing_Hour.csv"),
    response = "cnt",
    preprocess = function(dat) {
      # Drop non-predictive columns: instant (row ID), dteday (date string),
      # casual and registered (they sum to cnt, would leak the response)
      drop_cols <- c("instant", "dteday", "casual", "registered")
      dat <- dat[, !names(dat) %in% drop_cols, drop = FALSE]
      dat
    },
    pdp_features = NULL
  )
)

# Experiment toggles / settings (keep changes centralised here)
RUN_REPEATED_SPLITS <- TRUE
N_REPEATS <- 30
N_REPEATS_LARGE <- 15  # for large datasets (n > 10000 or p > 50)

N_FOLDS_GLMNET <- 10
N_FOLDS_RF <- 5

RF_DEFAULTS <- list(
  num_trees = 1000,
  mtry = 6,
  min_node_size = 1
)

POLY_MAX_DEGREE <- 15

REQUIRED_PACKAGES <- c(
  "readr",
  "glmnet",
  "rpart",
  "rpart.plot",
  "ranger",
  "pdp"
)

ensure_dirs <- function() {
  dirs <- c(R_DIR, FIG_DIR, OUTPUT_DIR, DATA_DIR)
  for (d in dirs) {
    if (!dir.exists(d)) dir.create(d, recursive = TRUE)
  }
  invisible(dirs)
}

check_packages <- function(pkgs = REQUIRED_PACKAGES) {
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop(
      "Missing required package(s): ",
      paste(missing, collapse = ", "),
      ". Install them before running main.R"
    )
  }
  invisible(TRUE)
}

clean_names <- function(x) {
  x <- gsub("\\(.*?\\)", "", x)
  x <- gsub("[^A-Za-z0-9]+", "_", x)
  x <- gsub("_+$", "", x)
  x <- gsub("^_+", "", x)
  tolower(x)
}

rmse <- function(actual, pred) {
  ok <- is.finite(actual) & is.finite(pred)
  actual <- actual[ok]
  pred <- pred[ok]
  if (length(actual) == 0) return(NA_real_)
  sqrt(mean((actual - pred)^2))
}

mae <- function(actual, pred) {
  ok <- is.finite(actual) & is.finite(pred)
  actual <- actual[ok]
  pred <- pred[ok]
  if (length(actual) == 0) return(NA_real_)
  mean(abs(actual - pred))
}

r2_score <- function(actual, pred) {
  ok <- is.finite(actual) & is.finite(pred)
  actual <- actual[ok]
  pred <- pred[ok]
  if (length(actual) == 0) return(NA_real_)
  ss_res <- sum((actual - pred)^2)
  ss_tot <- sum((actual - mean(actual))^2)
  if (ss_tot <= 0) return(NA_real_)
  1 - ss_res / ss_tot
}

calc_metrics <- function(actual, pred) {
  c(
    RMSE = rmse(actual, pred),
    MAE = mae(actual, pred),
    R2 = r2_score(actual, pred)
  )
}

assert_response <- function(df, response = RESPONSE) {
  if (!response %in% names(df)) {
    stop("Response column '", response, "' not found in data.")
  }
  invisible(TRUE)
}

as_metric_row <- function(model_name, metrics) {
  data.frame(
    Model = model_name,
    RMSE = unname(metrics["RMSE"]),
    MAE = unname(metrics["MAE"]),
    R2 = unname(metrics["R2"]),
    row.names = NULL,
    check.names = FALSE
  )
}