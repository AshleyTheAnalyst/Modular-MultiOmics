# Define Python Optuna objective
reticulate::py_run_string("
import optuna
import time
from datetime import datetime

print('=== VAE Hyperparameter Tuning Started ===')
start_total = time.time()
print('Start timestamp:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print('Total trials: 180 (18 combos × 10 cycles)\\n')

def objective(trial):
    start_time = time.time()
    
    latent_dim = trial.suggest_categorical('latent_dim', [15, 19, 20])
    # neurons1   = trial.suggest_categorical('neurons1', [64, 32, 16])
    # neurons2   = trial.suggest_categorical('neurons2', [64, 32, 16])
    batch_size = trial.suggest_categorical('batch_size', [32, 16])
    beta = trial.suggest_categorical('beta', [0.01, 0.001])
    
    res = r.train_vae_py(
        latent_dim = latent_dim,
        neurons1   = 64,
        neurons2   = 32,
        batch_size = batch_size,
        epochs     = 200,
        beta       = beta
    )
    
    train_rmse = float(res['train_rmse'][0])
    val_rmse   = float(res['val_rmse'][0])
    
    trial_time = time.time() - start_time
    print(f'[{trial.number}]train-rmse:{train_rmse:.6f} test-rmse:{val_rmse:.6f} '
          f'(took {trial_time:.1f}s; ld={latent_dim}, bs={batch_size}, beta={beta})')
    
    return val_rmse

study = optuna.create_study(direction=\"minimize\")
study.optimize(objective, n_trials=60)

# CREATE results_df (your original style)
trials_df = study.trials_dataframe()
trials_df['rank'] = trials_df['value'].rank(method='min', ascending=True)
results_df = trials_df.sort_values('value').reset_index(drop=True)
results_df = results_df[['number', 'value', 'rank', 'latent_dim', 'batch_size', 'beta']]

total_time = time.time() - start_total

print('\\n=== TUNING COMPLETE ===')
print('End timestamp:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print('BEST PARAMS:', study.best_params)
print('BEST VAL_RMSE:', study.best_value)
print(f'Total runtime: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes, {total_time:.0f} seconds)')
print('\\nTOP 5 RESULTS:')
print(results_df.head().to_string(index=False))

# RETURN TO R as DataFrame
r.results_df <- results_df
r.best_params <- study.best_params
r.best_value <- study.best_value
", local = TRUE, convert = TRUE)
