# Multi-Omics Integration with Modular VAE

**A flexible, reproducible computational pipeline for multi-omics data integration, missing value imputation, and latent space analysis**

[![R](https://img.shields.io/badge/R-4.0+-blue.svg)](https://www.r-project.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

This repository provides a **modular, end-to-end computational pipeline** for integrating heterogeneous molecular data using **Variational Autoencoders (VAEs)**. The framework is designed for:

- **Multi-modal data integration** with modality-specific encoders
- **Missing value imputation** leveraging learned latent representations
- **Comprehensive evaluation** including reconstruction quality, clustering validation, and latent space visualization
- **Reproducible training** with repeated runs and automated hyperparameter tuning

The pipeline was developed for analyzing large-scale clinical data from cohorts but is easily adaptable to other high-dimensional molecular datasets (transcriptomics, proteomics, etc.).

---

## 🔬 Key Features

### 1. Multi-Modal VAE Architecture
- **Modality-specific encoders** for integrating diverse data types
- **Shared latent space** enabling joint representation learning across modalities
- Configurable network architecture (hidden layers, activation functions, β-ELBO)

### 2. Missing Data Handling
- `create_missing_data()` – Controlled missingness simulation for benchmarking
- `imp_vae()` – VAE-based imputation leveraging learned latent representations
- Comparison against mean imputation baselines

### 3. Robust Training Framework
- `train_vae_repeated()` – Multiple training cycles (n_cycles = 10) with different random seeds for stability assessment
- **Early stopping** and **learning rate scheduling** via Keras callbacks
- **Stratified data splitting** to prevent experimental batch bias

### 4. Automated Hyperparameter Optimization
- Integration with **Optuna** (Python) via `reticulate`
- Tunes latent dimension, batch size, and β value
- Automatic cross-validation and results aggregation

### 5. Comprehensive Evaluation Suite

| Category | Methods |
|----------|---------|
| **Training** | Loss curves, β-KL vs reconstruction contribution |
| **Latent Space** | PCA, UMAP visualizations |
| **Reconstruction** | Per-gene scatter plots, MSE histograms (per-sample and per-feature) |
| **Imputation** | KDE distribution comparison, scatter plots for imputed sets |
| **Clustering** | Silhouette analysis, optimal k selection, gap statistics |
| **Interpretation** | Correlation heatmaps (latent dims vs. clinical variables), ridge plots |

---
## 🚀 Quick Start

### Prerequisites

- **R** (≥ 4.0) with packages: `keras`, `tensorflow`, `tidyverse`, `reticulate`, `future`, `furrr`, `caret`, `FactoMineR`, `cluster`
- **Python** (≥ 3.8) with: `tensorflow`, `keras`, `optuna`

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/VAE-MultiOmics-Integration.git
cd VAE-MultiOmics-Integration

# Install R dependencies
Rscript requirements.R

# Set up Python environment (recommended)
conda env create -f environment.yml
conda activate vae-env
```
---
## Basic Usage Example
```r

# Source the core functions
source("R/data_prep.R")
source("R/vae_core.R")
source("R/train_vae.R")
source("R/evaluation.R")

# 1. Prepare data with stratified splitting
split_result <- stratified_train_valid_imp_split(
  df = normalized_data,
  strata_vars = c("Gender", "AgeGroup", "stimulus_id"),
  numeric_cols = selected_features,
  train_prop = 0.7
)

# 2. Train VAE
result <- train_vae(
  latent_dim = 16,
  hidden1_units = 64,
  hidden2_units = 32,
  beta = 1e-4,
  batch_size = 32,
  epochs = 200,
  train_data = split_result$train_matrix,
  valid_data = split_result$valid_matrix,
  activation_arg1 = "leakyrelu",
  activation_arg2 = "linear"
)

# 3. Extract latent representations
latent_vectors <- result$latent_model %>% 
  predict(rbind(split_result$train_matrix, split_result$valid_matrix))

# 4. Visualize latent space (UMAP)
umap_plot <- generate_umap_plot(
  latent_vectors, 
  metadata = full_dataGp,
  color_var = "stimulus_id"
)

# 5. Evaluate reconstruction quality
recon_plot <- plot_reconstruction_scatter(
  vae_result = result,
  original_data = full_data,
  top_genes = c("gene1", "gene2", "gene3")
)
```
---
## 📊 Example Results
Loss Curves
[Insert loss curve figure here]
Loss curves showing training and validation loss across epochs. Validation loss (orange) closely tracks training loss (blue), indicating minimal overfitting.

Latent Space Visualization
[Insert UMAP figure here]
UMAP projection of the learned latent space, colored by experimental condition. The model captures distinct biological states without overfitting to technical factors.

Reconstruction Quality
[Insert scatter plot figure here]
Original vs. reconstructed expression values for key genes. High Spearman correlation demonstrates faithful reconstruction.

Missing Value Imputation
[Insert KDE figure here]
Kernel density estimation comparing original vs. VAE-imputed distributions. The imputed distribution closely matches the original, validating the imputation approach.

---
## 📝 Citation
If you use this framework in your research, please cite:

Li, A. (2026). A modular VAE framework for multi-omics integration and missing value imputation.
GitHub repository: https://github.com/AshleyTheAnalyst/Modular-MultiOmics

---
## 📄 License
This project is licensed under the MIT License – see the LICENSE file for details.


