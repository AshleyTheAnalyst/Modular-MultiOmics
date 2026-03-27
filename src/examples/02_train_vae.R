library(yaml)
params <- yaml::read_yaml("config/params.yaml")

latent_dims <- c(3,4,5, 30, 60, 90)

future::plan(multisession, workers = 11) # parallel but without console messages

start_time <- Sys.time()

# Run for all latent dimensions with 10 cycles
vae_train_rlt <- furrr::future_map(latent_dims, ~train_vae_repeated(
  latent_dim = .x,
  n_cycles = 10,
  hidden1_units = 64,
  hidden2_units = 32,
  beta = 1e-04,
  lr = 0.001,
  batch_size = 32,
  epochs = 200,
  activation_arg1 = "leakyrelu",
  activation_arg2 = "linear",
  train_data = train_data,
  valid_data = valid_data
),
.options = furrr_options(seed = TRUE),  # ← THIS FIXES WARNINGS
.progress = TRUE)

plan(sequential)  # Reset

total_time <- Sys.time() - start_time

message("TOTAL TIME: ", round(total_time, 1), " (", 
        round(as.numeric(total_time)/60, 1), " minutes)")

names(vae_train_rlt) <- paste0("latent_dim_", latent_dims)
