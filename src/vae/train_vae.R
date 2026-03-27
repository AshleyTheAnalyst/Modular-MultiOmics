# Training function
train_vae <- function(latent_dim, hidden1_units = 64, hidden2_units = 32, beta = 1e-04, lr = 0.001,
                      batch_size = 16, epochs = 200, 
                      activation_arg1, activation_arg2, train_data, valid_data, seed) {
  set.seed(seed)
  
  vae_obj <- build_vae(ncol(train_data), latent_dim, hidden1_units, hidden2_units, beta,
                       lr, activation_arg1, activation_arg2)
  
  # Train with progress message
  message("Training: latent_dim=", latent_dim, " h1=", hidden1_units, 
          " h2=", hidden2_units, " beta=", beta, " bs=", batch_size)
  
  history <- vae_obj$model %>% fit(
    x = train_data, 
    y = train_data,
    validation_data = list(valid_data, valid_data),
    epochs = epochs, 
    batch_size = batch_size,
    callbacks = list(
      callback_early_stopping(monitor = "val_loss", patience = 10, restore_best_weights = TRUE),
      callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5, patience = 5)
    ),
    verbose = 0
  )
  
  # Completion message
  epochs_completed <- length(history$metrics$loss)
  
  message("Completed ", epochs_completed, " epochs for latent_dim = ", latent_dim)
  message("Final training loss: ", round(tail(history$metrics$loss, 1), 5))
  message("Final validation loss: ", round(tail(history$metrics$val_loss, 1), 5))
  
  latent_values <- vae_obj$latent_model %>% predict(rbind(train_data, valid_data))
  reconstructed_values <- vae_obj$model %>% predict(rbind(train_data, valid_data))
  
  list(
    latent_dim = latent_dim,
    # vae_model = vae_obj$model,
    vae_obj = vae_obj,  # keep full object for imp_vae()
    latent_values = latent_values,
    reconstructed_values = reconstructed_values,
    history = history
  )
}

# Define function to train VAE for multiple cycles and average results
train_vae_repeated <- function(latent_dim, n_cycles = 10, hidden1_units = 64, hidden2_units = 32, beta = 1e-04, lr = 0.001,
                               batch_size = 16, epochs = 200, 
                               activation_arg1, activation_arg2, train_data, valid_data) {
  
  # Store results from each cycle
  all_histories <- list()
  all_latent_values <- list()
  all_reconstructed_values <- list()
  all_final_train_loss <- c()
  all_final_val_loss <- c()
  all_epochs_completed <- c()
  
  message("\n========== Training latent_dim = ", latent_dim, " for ", n_cycles, " cycles ==========")
  
  for (cycle in 1:n_cycles) {
    set.seed(1234 + cycle)  # Different seed for each cycle
    message("\n--- Cycle ", cycle, "/", n_cycles, " ---")
    
    # Train VAE
    result <- train_vae(
      latent_dim = latent_dim,
      hidden1_units = hidden1_units,
      hidden2_units = hidden2_units,
      beta = beta,
      lr = lr,
      batch_size = batch_size,
      epochs = epochs,
      activation_arg1 = activation_arg1,
      activation_arg2 = activation_arg2,
      train_data = train_data,
      valid_data = valid_data,
      seed = 1234 + cycle 
    )
    
    # Store results
    all_histories[[cycle]] <- result$history
    all_latent_values[[cycle]] <- result$latent_values
    all_reconstructed_values[[cycle]] <- result$reconstructed_values
    all_final_train_loss[cycle] <- tail(result$history$metrics$loss, 1)
    all_final_val_loss[cycle] <- tail(result$history$metrics$val_loss, 1)
    all_epochs_completed[cycle] <- length(result$history$metrics$loss)
  }
  
  # --- Average the results ---
  
  # 1. Average the loss curves (need to align by epoch)
  # Get max epochs across all cycles
  max_epochs <- max(sapply(all_histories, function(h) length(h$metrics$loss)))
  
  # Create matrices for train/val loss
  train_loss_matrix <- matrix(NA, nrow = max_epochs, ncol = n_cycles)
  val_loss_matrix <- matrix(NA, nrow = max_epochs, ncol = n_cycles)
  
  for (cycle in 1:n_cycles) {
    epochs_cycle <- length(all_histories[[cycle]]$metrics$loss)
    train_loss_matrix[1:epochs_cycle, cycle] <- all_histories[[cycle]]$metrics$loss
    val_loss_matrix[1:epochs_cycle, cycle] <- all_histories[[cycle]]$metrics$val_loss
  }
  
  # Calculate mean and sd across cycles
  avg_train_loss <- rowMeans(train_loss_matrix, na.rm = TRUE)
  avg_val_loss <- rowMeans(val_loss_matrix, na.rm = TRUE)
  
  # 2. Average latent values and reconstructed values
  avg_latent_values <- Reduce(`+`, all_latent_values) / n_cycles
  avg_reconstructed_values <- Reduce(`+`, all_reconstructed_values) / n_cycles
  
  # 3. Average final metrics
  avg_final_train_loss <- mean(all_final_train_loss)
  avg_final_val_loss <- mean(all_final_val_loss)
  avg_epochs_completed <- mean(all_epochs_completed)
  
  # Create averaged history object
  avg_history <- list(
    metrics = list(
      loss = avg_train_loss,
      val_loss = avg_val_loss
    ),
    params = all_histories[[1]]$params
  )
  class(avg_history) <- "keras_training_history"
  
  # Return averaged results
  list(
    latent_dim = latent_dim,
    vae_obj = result$vae_obj,  # Keep the last VAE object
    latent_values = avg_latent_values,
    reconstructed_values = avg_reconstructed_values,
    history = avg_history,
    # Raw data for reference
    all_histories = all_histories,
    all_final_train_loss = all_final_train_loss,
    all_final_val_loss = all_final_val_loss,
    avg_final_train_loss = avg_final_train_loss,
    avg_final_val_loss = avg_final_val_loss,
    avg_epochs_completed = avg_epochs_completed,
    n_cycles = n_cycles
  )
}


train_vae_py <- function(latent_dim, neurons1, neurons2, batch_size, epochs, beta, lr = 0.001) {
  result <- train_vae(
    latent_dim = latent_dim,
    hidden1_units = neurons1,
    hidden2_units = neurons2,
    beta = beta,
    lr = lr,
    batch_size = batch_size,
    epochs = epochs,
    activation_arg1 = "leakyrelu",
    activation_arg2 = "linear",
    train_data = train_data,
    valid_data = valid_data,
    seed = 1234
  )
  
  data.frame(
    latent_dim = latent_dim,
    neurons1 = neurons1,
    neurons2 = neurons2,
    batch_size = batch_size,
    epochs = epochs,
    beta = beta,
    train_rmse = sqrt(tail(result$history$metrics$loss, 1)),
    val_rmse   = sqrt(tail(result$history$metrics$val_loss, 1))
  )
}

