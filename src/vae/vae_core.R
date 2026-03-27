# Sampling function
sampling_layer <- function(latent_dim) {
  function(args) {
    z_mean <- args[, 1:latent_dim]
    z_log_var <- args[, (latent_dim + 1):(2 * latent_dim)]
    epsilon <- k_random_normal(shape = k_shape(z_mean), mean = 0, stddev = 1)
    z_mean + k_exp(z_log_var / 2) * epsilon
  }
}


# build_vae function
build_vae <- function(input_dim, latent_dim, hidden1_units = 64, hidden2_units = 32, beta = 1e-04,
                      lr = 0.001, activation_arg1, activation_arg2) {
  
  # Encoder
  encoder_input <- layer_input(shape = input_dim)
  x <- encoder_input %>%
    layer_dense(units = hidden1_units)
  
  # Conditional activation function
  if (activation_arg1 == "leakyrelu") {
    x <- x %>% layer_activation_leaky_relu(alpha = 0.01)
  } else {
    x <- x %>% layer_activation(activation_arg1)
  }
  
  x <- x %>%
    layer_dense(units = hidden2_units)
  
  if (activation_arg1 == "leakyrelu") {
    x <- x %>% layer_activation_leaky_relu(alpha = 0.01)
  } else {
    x <- x %>% layer_activation(activation_arg1)
  }
  
  encoder_output <- x %>% layer_dense(units = latent_dim * 2)  # mean + log_var
  
  z <- layer_lambda(f = sampling_layer(latent_dim))(encoder_output)
  
  # Encoder model
  encoder <- keras_model(encoder_input, encoder_output)
  latent_model <- keras_model(encoder_input, z)
  
  # Decoder (same logic)
  decoder_input <- layer_input(shape = latent_dim)
  x <- decoder_input %>% layer_dense(units = hidden2_units)
  
  if (activation_arg1 == "leakyrelu") {
    x <- x %>% layer_activation_leaky_relu(alpha = 0.01)
  } else {
    x <- x %>% layer_activation(activation_arg1)
  }
  
  x <- x %>% layer_dense(units = hidden1_units)
  
  if (activation_arg1 == "leakyrelu") {
    x <- x %>% layer_activation_leaky_relu(alpha = 0.01)
  } else {
    x <- x %>% layer_activation(activation_arg1)
  }
  
  decoder_output <- x %>% layer_dense(units = input_dim, activation = activation_arg2)
  
  decoder <- keras_model(decoder_input, decoder_output)
  
  # VAE model
  vae_output <- decoder(z)
  vae <- keras_model(encoder_input, vae_output)
  
  # VAE loss function
  vae_loss <- function(x, x_decoded) {
    encoder_out <- encoder(x)
    z_mean <- encoder_out[, 1:latent_dim]
    z_log_var <- encoder_out[, (latent_dim + 1):(2 * latent_dim)]
    
    # Reconstruction loss (only on observed values)
    # mask <- k_cast(k_not_equal(x, -1), dtype = "float32")
    recon_loss <- k_mean(k_square(x - x_decoded))
    
    # KL divergence
    kl_loss <- -0.5 * k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), 
                             axis = -1L)
    
    
    # beta <- 1e-04
    recon_loss +  beta * kl_loss
  }
  
  # ---------- NEW metrics recon + KL per epoch ----------
  recon_metric <- function(x, x_decoded) {
    k_mean(k_square(x - x_decoded))
  }
  
  kl_metric <- function(x, x_decoded) {
    encoder_out <- encoder(x)
    z_mean <- encoder_out[, 1:latent_dim]
    z_log_var <- encoder_out[, (latent_dim + 1):(2 * latent_dim)]
    
    -0.5 * k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var),
                  axis = -1L)
  }
  
  vae %>% compile(
    optimizer = optimizer_adam(learning_rate = lr),
    loss = vae_loss
  )
  
  list(
    model = vae,
    encoder = encoder,
    decoder = decoder,
    latent_model = latent_model, # Return latent vector model
    latent_dim = latent_dim
  )
}
