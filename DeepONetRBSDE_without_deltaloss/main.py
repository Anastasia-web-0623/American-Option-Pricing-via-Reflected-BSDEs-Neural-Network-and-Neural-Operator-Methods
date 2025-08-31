import jax
import jax.numpy as jnp
import numpy as np
import time

from deep_config import RBSDEConfig
from rbsde_solver import RBSDESolver
from deep_plots import save_training_plots, save_premium_surfaces, save_validation_plots, save_delta_plots

def main():

    print("RBSDE-DeepONet Solver (Normalized by S0) ")
    

    start_time = time.time()
    key = jax.random.PRNGKey(42)
    config = RBSDEConfig()

    print(f"\nConfiguration:")
    print(f"  Strike Price K: {config.K}")
    print(f"  Training Iterations: {config.n_iterations}")
    print(f"  Loss weights: ITM={config.itm_weight}, ATM={config.atm_weight} (range={config.atm_range}), OTM={config.otm_weight}")
    print(f"  RBSDE Weights: step={config.rbsde_step_weight}, global={config.rbsde_global_weight}")
    
    solver = RBSDESolver(config)
    end_init_time = time.time()
    print(f"Initialization time: {end_init_time - start_time:.2f} seconds")

    # training model
    print("\nStarting training")
    start_train_time = time.time()
    trained_networks, training_info = solver.train(key)
    end_train_time = time.time()
    print(f"\nTraining complete! Final loss: {training_info['final_loss']:.6f}")
    print(f"Training time: {end_train_time - start_train_time:.2f} seconds")
    
    # evulate the model
    print("\nStarting evaluation")
    start_eval_time = time.time()
    test_key = jax.random.PRNGKey(999)
    results = solver.evaluate(trained_networks, test_key, 400)
    end_eval_time = time.time()
    print(f"Evaluation complete.")
    print(f"Evaluation time: {end_eval_time - start_eval_time:.2f} seconds")
    
    stats = results['statistics']
    nn = stats['non_normalized']
    nm = stats['normalized']

    print("\n" + "="*60)
    print(" Evaluation Results (non-normalized)")
    print("="*60)
    print(f"Price Metrics:")
    print(f"  Mean Absolute Error (MAE):      {nn['price']['mean_abs_err']:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {nn['price']['rmse']:.4f}")
    mse = nn['price']['rmse'] ** 2
    print(f"  Mean Squared Error (MSE):       {mse:.4f}")
    print(f"  Price Correlation:              {nn['price']['correlation']:.4f}")
    print(f"  Mean Rel Error vs BS (>0.1):    {nn['price']['mean_rel_err_vs_bs_%_filtered']:.2f}%")
    print(f"  sMAPE vs BS:                    {nn['price']['smape_vs_bs_%']:.2f}%")
    print("-" * 30)
    print(f"Delta Metrics (Δ = Z/(σS)):")
    print(f"  Delta RMSE:                     {nn['delta']['rmse']:.4f}")
    print(f"  Delta Corr (clipped):           {nn['delta']['corr_clipped']:.4f}")
    print(f"  Delta Corr (unclipped):         {nn['delta']['corr_unclipped']:.4f}")
    print(f"Z Metrics (Z = Δ·σ·S):")
    print(f"  Z RMSE:                         {stats['non_normalized']['Z']['rmse']:.4f}")
    print(f"  Z Correlation:                  {stats['non_normalized']['Z']['correlation']:.4f}")
    print("\nAmerican Option Characteristics:")
    print(f"  Mean American Premium:          {nn['american_premium_mean']:.4f}")

    # plot the figures
    print("\nStarting plot generation...")
    start_plot_time = time.time()
    save_training_plots(training_info)
    save_premium_surfaces(
        solver,
        trained_networks,
        S0_range=config.S_lim,
        sigma_range=config.sigma_lim,
        r_range=config.r_lim,
        n_S0=80, n_sigma=80, n_r=80,
        r_fixed=0.05,
        sigma_fixed=0.30,
    )
    save_validation_plots(results)
    save_delta_plots(results, solver, trained_networks, r_fixed=0.05)
    end_plot_time = time.time()
    print(f"Plot generation time: {end_plot_time - start_plot_time:.2f} seconds")
    
    print("\nTotal execution time: {0:.2f} seconds".format(end_plot_time - start_time))

    print("\nSaved figures:")
    print("  - training_loss.png")
    print("  - loss_components.png")
    print("  - premium_surface_r_fixed.png")
    print("  - premium_surface_sigma_fixed.png")
    print("  - premium_heatmap_r_fixed.png")
    print("  - premium_heatmap_sigma_fixed.png")
    print("  - price_parity.png")
    print("  - price_parity_normalized.png")
    print("  - price_residual_hist.png")
    print("  - price_residual_hist_normalized.png")
    print("  - rel_error_vs_S0.png")
    print("  - rel_error_vs_sigma.png")
    print("  - rel_error_vs_r.png")
    print("  - delta_parity.png")
    print("  - Z_parity.png")
    print("  - delta_error_hist.png")
    print("  - delta_vs_S0_sigma_groups.png")
    print("  - delta_heatmap_r_fixed.png")

if __name__ == "__main__":
    main()