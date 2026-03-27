# Install ALL required packages
required_packages <- c(
  "magrittr", "umap", "ggplot2", "patchwork",     
  "dplyr", "tidyr", "caret", "recipes",           
  "FactoMineR", "factoextra",                     
  "pheatmap", "ggridges", "cluster", "corrplot",
  "future", "furrr", "future.callr"
)

# Install missing packages
missing <- required_packages[!required_packages %in% installed.packages()[,"Package"]]
if(length(missing)) install.packages(missing, dependencies = TRUE)

library(keras)
library(magrittr)  # for pipe operator %>%
library(umap)
library(ggplot2)
library(patchwork)
library(dplyr)
library(caret)
library(recipes)
library(tidyverse)
library("FactoMineR")
library("factoextra")
library(pheatmap)
library(ggridges)
library(cluster)
library("corrplot")
library(future)        # plan()
library(furrr)         # future_map()
library(future.callr)


# Use the r-tensorflow-new environment
use_virtualenv("r-tensorflow-new")

# Verify installation
reticulate::py_run_string("
import numpy, tensorflow as tf, keras
print(f'NumPy version: {numpy.__version__}')
print(f'TensorFlow: {tf.__version__}')
print(f'Keras: {keras.__version__}')
")

######################### data import #################################
syn_data <- rio::import("data/syn_data.csv") 

anyNA(syn_data) 

selected_cols <- colnames(real_data[,-1:-9])

length(selected_cols)


######################### normalization #################################
# reorder Index (get rid of enrollment date bias)
set.seed(123)

syn_data_shuffle <- syn_data[sample(nrow(syn_data)), ]

original_means <- colMeans(syn_data_shuffle[, selected_cols], na.rm = TRUE)
original_sds <- apply(syn_data_shuffle[, selected_cols], 2, sd, na.rm = TRUE)

# min-max normalization 
syn_data_min_max <- syn_data_shuffle |> 
  # min-max normalization 
  mutate(across(where(is.numeric), ~ (. - min(., na.rm = TRUE)) / (max(., na.rm = TRUE) - min(., na.rm = TRUE))))

# standardization
syn_data_std <- syn_data_shuffle |> 
  # min-max normalization 
  mutate(across(where(is.numeric), ~ (. - mean(., na.rm = TRUE)) / sd(., na.rm = TRUE)))

######################### split data #################################
strata_columns <- c('Gender', 'AgeGroup')

selected_cols

# 1. Split data (ONCE for all latent dims) change the 
split_result <- stratified_train_valid_imp_split(real_data_std, 
                                                 strata_vars = strata_columns, 
                                                 numeric_cols = selected_cols, 
                                                 train_prop = 0.7, seed = 1234)
train_data <- split_result$train_matrix
valid_data <- split_result$valid_matrix
train_dataGp <- split_result$train_df
valid_dataGp <- split_result$valid_df
imp_dataGp <- split_result$imp_df
imp_data <- split_result$imp_matrix
train_means <- colMeans(train_data)

# FULL DATA for visualization (train + val)
full_data <- rbind(train_data, valid_data)  #  all samples with colnames
full_dataGp <- rbind(train_dataGp, valid_dataGp)  # Metadata: all samples with colnames and HHKID info

# FULL FULL data (train + val + imp)
complete_data <- rbind(train_data, valid_data, imp_data)   

complete_dataGp <- rbind(
  train_dataGp %>% mutate(dataset = "train"),
  valid_dataGp %>% mutate(dataset = "valid"),
  imp_dataGp %>% mutate(dataset = "imp")
) 
