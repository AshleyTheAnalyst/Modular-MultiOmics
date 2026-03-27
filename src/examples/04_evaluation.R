combined_df <- combine_results(vae_train_rlt)

# Plot latent dimension
combined_plot<- plot_training_validation_loss(
  data = combined_df,
  latent_dims = latent_dims,
  title = "VAE Loss: Latent Dimension 3-5",
  subtitle = "z-score normalization"
)

combined_plot

# summary of final loss function (single run)
# Extract final losses for all latent dimensions
loss_df <- purrr::map_dfr(names(vae_train_rlt), function(name) {
  hist <- vae_train_rlt[[name]]$history$metrics
  data.frame(
    model = name,
    final_train_loss = tail(hist$loss, 1),
    final_val_loss = tail(hist$val_loss, 1)
  )
})

print(loss_df)

# --------------------------------- KL stacked plot -------------------------------
