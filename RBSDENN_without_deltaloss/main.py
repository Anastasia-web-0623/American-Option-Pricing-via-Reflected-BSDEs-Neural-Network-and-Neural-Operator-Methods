import os
import time
import json
import random
import time
import math
import torch
import pandas as pd
import numpy as np

from rbsde_model import AmericanCallRBSDE
from utils import (
    evaluate_vs_bs_both,
    sample_cases_aligned,
    ensure_dir,
    black_scholes_call_price
)
from visualization import plot_and_save
# Helper function to evaluate the model on a single batch
def evaluate_model(model: AmericanCallRBSDE):
    model.net.eval()
    with torch.no_grad():
        test_batch_size = 8000
        payoff_T_norm, _, y_path, _, _ = model.rbsde_forward(test_batch_size)
        terminal_mse_norm = torch.mean((payoff_T_norm - y_path[-1]) ** 2).item()
        
        # push for avg_total_push
        _, _, _, push_path, _ = model.rbsde_forward(test_batch_size)
        total_push = sum(p.sum().item() for p in push_path)
        avg_total_push_norm = total_push / test_batch_size
        
        price_abs, price_norm = model.get_option_price()
        
        return price_abs, price_norm, terminal_mse_norm, avg_total_push_norm

def main():
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Setup output directory
    ts = time.strftime('%Y%m%d-%H%M%S')
    outdir = os.path.join('./results', f'final_no_delta_loss_{ts}')
    ensure_dir(outdir)
    
    # Generate a list of test cases
    cases = sample_cases_aligned(
        S_range=(1.0, 4.0),
        r_lim=(0.02, 0.08),
        sigma_lim=(0.15, 0.45),
        K_abs=2.0,
        T_fixed=1.0,
        n_cases=30,
        atm_range=0.1,
    )
    
    # Training parameters
    N_STEPS = 160
    DEVICE = 'cpu'
    LR = 8e-4
    EPOCHS = 2000
    BATCH = 384
    
    rows = []
    for idx, cfg in enumerate(cases, 1):
        print(f"\n=== Option {idx}/{len(cases)} ({cfg['bucket']}) ===")
        print(json.dumps({k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in cfg.items()}, indent=2))
        
        # Initialize and train the model for each case
        model = AmericanCallRBSDE(
            S0=cfg['S0'], K=cfg['K'], r=cfg['r'], q=cfg['q'], sigma=cfg['sigma'], 
            T=cfg['T'], n_steps=N_STEPS, device=DEVICE, use_antithetic=True
        )
        
        # 记录训练开始时间
        start_time_train = time.time()
        
        model.train(
            n_epochs=EPOCHS, batch_size=BATCH, lr=LR, 
            verbose_every=250, patience=5000
        )
        
        # 记录训练结束时间并计算时长
        end_time_train = time.time()
        training_duration = end_time_train - start_time_train
        
        # 每次结束一组时输出时间
        print(f"training time: {training_duration:.2f} second")
        
        # Evaluate and collect results
        am_abs, am_norm, term_mse, avg_push = evaluate_model(model)
        pred = model.predict_t0()
        dn_delta = pred['delta']
        bs_abs = pred['bs_price']
        bs_delta = pred['bs_delta']
        eu_norm = bs_abs / cfg['S0']
        premium_norm = am_norm - eu_norm

        # Calculate y0 target from a larger batch of paths
        with torch.no_grad():
            payoff_T_norm, _, y_path, push_path, _ = model.rbsde_forward(8000)
            disc_T = math.exp(-model.r * model.T)
            disc_push = 0.0
            for i, push in enumerate(push_path):
                t_next = (i + 1) * model.dt
                disc_push += math.exp(-model.r * t_next) * push.mean().item()
            y0_target = disc_T * payoff_T_norm.mean().item() + disc_push
            y0_gap = abs(am_norm - y0_target)

        row = dict(
            idx=idx,
            bucket=cfg['bucket'],
            moneyness=cfg['moneyness'],
            S0=cfg['S0'],
            K=cfg['K'],
            r=cfg['r'],
            q=cfg['q'],
            sigma=cfg['sigma'],
            T=cfg['T'],
            n_steps=N_STEPS,
            epochs=EPOCHS,
            training_duration_sec=training_duration,  
            Am_Price=am_abs,
            Am_Price_norm=am_norm,
            Eu_Price=bs_abs,
            Eu_Price_norm=eu_norm,
            Am_Premium_norm=premium_norm,
            Am_Delta=dn_delta,
            BS_Delta=bs_delta,
            Terminal_MSE_norm=term_mse,
            Avg_Total_Push_norm=avg_push,
            y0_pred_norm=am_norm,
            y0_target_norm=y0_target,
            y0_gap_norm=y0_gap
        )
        rows.append(row)
        pd.DataFrame([row]).to_csv(os.path.join(outdir, f'case_{idx:02d}.csv'), index=False)
        
    # Combine and save all results
    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, 'results_all.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved all results to: {csv_path}")

    # Generate and save plots
    plot_and_save(df, outdir)
    print(f"Figures saved to: {outdir}")

    # Compute and print summary statistics
    stats_both = evaluate_vs_bs_both(df, atm_range=0.1)
    nn = stats_both['non_normalized']['price']
    nd = stats_both['non_normalized']['delta']
    nm = stats_both['normalized']['price']

    print("\n" + "="*60)
    print(" Evaluation Results (Non-normalized: absolute price)")
    print("="*60)
    print(f" MAE: {nn['MAE']:}")
    print(f" MSE: {nn['MSE']:}")
    print(f" RMSE: {nn['RMSE']:}")
    print(f" Correlation: {nn['Correlation']:}")
    print(f" RelErr vs BS (all): {nn['RelErr_vs_BS_pct']:}%")
    print(f" RelErr vs BS (>0.1): {nn['RelErr_vs_BS_pct_filtered']:f}%")
    print(f" RelErr vs S0: {nn['RelErr_vs_S0_pct']:f}%")
    print(f" sMAPE: {nn['sMAPE_pct']:.2f}%")
    print("-"*30)
    print(f" Delta MAE: {nd['MAE']:}")
    print(f" Delta MSE: {nd['MSE']:}")
    print(f" Delta RMSE: {nd['RMSE']:}")
    print(f" Delta Correlation: {nd['Correlation']:}")

    print("\n" + "="*60)
    print(" Evaluation Results (Normalized: price / S0)")
    print("="*60)
    print(f" MAE_norm: {nm['MAE']:}")
    print(f" MSE_norm: {nm['MSE']:}")
    print(f" RMSE_norm: {nm['RMSE']:}")
    print(f" Corr_norm: {nm['Correlation']:}")
    print(f" RelErr_norm vs BS (all): {nm['RelErr_vs_BS_pct']:}%")
    print(f" RelErr_norm vs BS (>0.1): {nm['RelErr_vs_BS_pct_filtered']:}%")
    print(f" RelErr_norm vs S0: {nm['RelErr_vs_S0_pct']:}% ")
    print(f" sMAPE_norm: {nm['sMAPE_pct']:}%")

    # Save summary metrics to CSV and JSON
    rows_summary = []
    rows_summary.append({'group':'non_normalized','target':'price', **nn})
    rows_summary.append({'group':'normalized','target':'price', **nm})
    rows_summary.append({'group':'non_normalized','target':'delta', **nd})

    df_sum = pd.DataFrame(rows_summary)
    sum_csv = os.path.join(outdir, 'final_metrics_summary.csv')
    sum_json = os.path.join(outdir, 'final_metrics_summary.json')
    df_sum.to_csv(sum_csv, index=False)
    with open(sum_json,'w') as f:
        json.dump(rows_summary, f, indent=2)
    print(f"\nSaved final metrics summary to:\n {sum_csv}\n {sum_json}")

    pb = stats_both['per_bucket']
    print("\nPer-bucket (non_normalized):")
    for b, m in pb['non_normalized'].items():
        print(f" {b}: count={m['count']}, MAE={m['MAE']:}, MSE={m['MSE']:}, RMSE={m['RMSE']:}")
    print("\nPer-bucket (normalized):")
    for b, m in pb['normalized'].items():
        print(f" {b}: count={m['count']}, MAE={m['MAE']:}, MSE={m['MSE']:}, RMSE={m['RMSE']:}")


if __name__ == '__main__':
    main()