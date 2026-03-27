imp_vae <- function(vae_result, train_data, valid_data, imp_data, train_means,
                    missing_cols, missing_rate) {
  
  # STEP 1: Create missing data
  current_missing_data <- create_missing_data(imp_data, missing_rate, missing_cols)
  
  # STEP 2: Pre-imputation with feature means  
  pre_imputed_data <- impute_with_means(current_missing_data, train_means)
  
  # STEP 3: Missing mask  latent_values_imp <- predict(vae_result$vae_obj$latent_model, pre_imputed_data)
  
  missing_mask <- is.na(current_missing_data[, selected_cols])
  
  # STEP 4: Get latent and reconstructed values for imputation set ONLY
  reconstructed_values_imp <- predict(vae_result$vae_obj$model, pre_imputed_data)
  
  # STEP 5: Full dataset predictions (train + val + imputed values)
  full_input_data <- rbind(train_data, valid_data, pre_imputed_data)
  full_latent_values <- predict(vae_result$vae_obj$latent_model, full_input_data)
  reconstructed_values <- predict(vae_result$vae_obj$model, full_input_data)
  
  
  list(
    missing_mask = missing_mask,
    # latent_values_imp = latent_values_imp,
    reconstructed_values_imp = reconstructed_values_imp,
    full_latent_values = full_latent_values,
    full_input_data = full_input_data,
    reconstructed_values = reconstructed_values
    
  )
}
