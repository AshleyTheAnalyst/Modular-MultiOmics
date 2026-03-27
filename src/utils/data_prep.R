library(tidyverse)

# function to introduce NA values
create_missing_data <- function(data, missing_rate, cols = NULL) {
  data_with_missing <- data
  if (is.null(cols)) {
    cols <- colnames(data_with_missing)
  }
  total_values <- nrow(data_with_missing) * length(cols)
  num_missing <- round(total_values * missing_rate)
  missing_indices <- sample(1:total_values, num_missing, replace = FALSE)
  data_vec <- as.vector(as.matrix(data_with_missing[, cols]))
  data_vec[missing_indices] <- NA  # keep NA for missing
  data_with_missing[, cols] <- matrix(data_vec, nrow = nrow(data_with_missing), ncol = length(cols))
  return(data_with_missing)
}


# function to stratified sampling + split data into training, testing, and imputation sets
stratified_train_valid_imp_split <- function(df, strata_vars, numeric_cols, train_prop = 0.7, seed=123) {
  library(caret)
  set.seed(seed)
  
  df <- df %>% mutate(strata = interaction(!!!rlang::syms(strata_vars), drop=TRUE))
  
  train_idx <- caret::createDataPartition(df$strata, p = train_prop, list = FALSE)
  train_df <- df[train_idx, ]
  val_df <- df[-train_idx, ]
  
  # split 30% validation set into half-half (15% + 15%)
  train_idx2 <- caret::createDataPartition(val_df$strata, p = 0.5, list = FALSE)
  valid_df <- val_df[train_idx2, ]
  imp_df <- val_df[-train_idx2, ]
  
  return(list(
    train_matrix = train_df %>% dplyr::select(all_of(numeric_cols)) %>% as.matrix(),
    valid_matrix = valid_df %>% dplyr::select(all_of(numeric_cols)) %>% as.matrix(),
    imp_matrix = imp_df %>% dplyr::select(all_of(numeric_cols)) %>% as.matrix(),
    full_matrix = df %>% dplyr::select(all_of(numeric_cols)) %>% as.matrix(),
    train_df = train_df,
    valid_df = valid_df,
    imp_df = imp_df,
    train_idx = train_idx,  # Add: Original indices for train
    train_idx2 = train_idx2  # Add: Indices for valid/imp in val_df
  ))
}


# function to replace NA into feature mean
impute_with_means <- function(data, means) {
  for (j in seq_along(means)) {
    na_indices <- is.na(data[, j])
    if (any(na_indices)) {
      data[na_indices, j] <- means[j]
    }
  }
  return(data)
}

