import os
import math
import time
import json
import random
import numpy as np
import pandas as pd
import torch
from typing import Tuple, Dict
from rbsde_solver import AmericanCallRBSDE
from visualization import plot_and_save, ensure_dir 

def evaluate_model(model: AmericanCallRBSDE) -> Tuple[float, float, float, float]:
    model.net.eval()
    with torch.no_grad():
        test_batch_size = 8000
        payoff_T_norm, _, y_path, _, _ = model.rbsde_forward(test_batch_size)
        terminal_mse_norm = torch.mean((payoff_T_norm - y_path[-1]) ** 2).item()
            
        _, _, _, push_path, _ = model.rbsde_forward(test_batch_size)
        total_push = sum(p.sum().item() for p in push_path)
        avg_total_push_norm = total_push / test_batch_size
            
        price_abs, price_norm = model.get_option_price()
            
        return price_abs, price_norm, terminal_mse_norm, avg_total_push_norm
def _core_price_metrics(p_dn: np.ndarray, p_bs: np.ndarray, S0: np.ndarray):
    eps = 1e-6
    abs_err = np.abs(p_dn - p_bs)
    mae = float(abs_err.mean())
    mse = float(((p_dn - p_bs) ** 2).mean())
    rmse = float(np.sqrt(mse))
    corr = float(np.corrcoef(p_dn, p_bs)[0, 1]) if p_dn.size > 1 else np.nan
    rel_bs = abs_err / np.clip(np.abs(p_bs), eps, None) * 100.0
    rel_bs_filtered = float(rel_bs[np.abs(p_bs) > 0.1].mean()) if np.any(np.abs(p_bs) > 0.1) else np.nan
    smape = float((2.0 * abs_err / np.clip(np.abs(p_dn) + np.abs(p_bs), eps, None)).mean() * 100.0)
    rel_s0 = float((abs_err / np.clip(np.abs(S0), eps, None)).mean() * 100.0)
    
    return dict(
        MAE=mae, MSE=mse, RMSE=rmse, Correlation=corr, 
        RelErr_vs_BS_pct=float(rel_bs.mean()), RelErr_vs_BS_pct_filtered=rel_bs_filtered, 
        sMAPE_pct=smape, RelErr_vs_S0_pct=rel_s0
    )

def evaluate_vs_bs_both(results_df: pd.DataFrame, atm_range: float = 0.1) -> dict:
    df = results_df.copy()
    S0 = df['S0'].to_numpy()
    K = df['K'].to_numpy()

    abs_metrics = _core_price_metrics(df['Am_Price'].to_numpy(), df['Eu_Price'].to_numpy(), S0)
    norm_metrics = _core_price_metrics(df['Am_Price_norm'].to_numpy(), df['Eu_Price_norm'].to_numpy(), S0)

    dd = df['Am_Delta'].to_numpy()
    bd = df['BS_Delta'].to_numpy()
    delta_mae = float(np.abs(dd - bd).mean())
    delta_mse = float(((dd - bd) ** 2).mean())
    delta_rmse = float(np.sqrt(delta_mse))
    delta_corr = float(np.corrcoef(dd, bd)[0, 1]) if dd.size > 1 else np.nan
    delta_metrics = dict(MAE=delta_mae, MSE=delta_mse, RMSE=delta_rmse, Correlation=delta_corr)

    itm = S0 > K
    atm = np.abs(S0 - K) <= atm_range
    otm = S0 < K

    def _bucket(mask, pdn, pbs):
        if not np.any(mask):
            return dict(count=0, MAE=np.nan, MSE=np.nan, RMSE=np.nan)
        e = pdn[mask] - pbs[mask]
        return dict(count=int(mask.sum()), MAE=float(np.abs(e).mean()), MSE=float((e**2).mean()), RMSE=float(np.sqrt((e**2).mean())))

    per_bucket = {
        'non_normalized': {
            'ITM': _bucket(itm, df['Am_Price'].to_numpy(), df['Eu_Price'].to_numpy()),
            'ATM': _bucket(atm, df['Am_Price'].to_numpy(), df['Eu_Price'].to_numpy()),
            'OTM': _bucket(otm, df['Am_Price'].to_numpy(), df['Eu_Price'].to_numpy()),
        },
        'normalized': {
            'ITM': _bucket(itm, df['Am_Price_norm'].to_numpy(), df['Eu_Price_norm'].to_numpy()),
            'ATM': _bucket(atm, df['Am_Price_norm'].to_numpy(), df['Eu_Price_norm'].to_numpy()),
            'OTM': _bucket(otm, df['Am_Price_norm'].to_numpy(), df['Eu_Price_norm'].to_numpy()),
        }
    }
    
    return {
        'non_normalized': {'price': abs_metrics, 'delta': delta_metrics},
        'normalized': {'price': norm_metrics},
        'per_bucket': per_bucket
    }

def sample_cases_aligned(
    S_range=(1.0, 4.0),
    r_lim=(0.02, 0.08),
    sigma_lim=(0.15, 0.45),
    K_abs=2.0,
    T_fixed=1.0,
    n_cases=30,
    atm_range=0.1,
):
    cases = []
    for _ in range(n_cases):
        S0 = float(np.random.uniform(*S_range))
        r = float(np.random.uniform(*r_lim))
        sigma = float(np.random.uniform(*sigma_lim))
        K = float(K_abs)
        T = float(T_fixed)
        q = 0.0
        if S0 > K:
            bucket = 'ITM'
        elif abs(S0 - K) <= atm_range:
            bucket = 'ATM'
        else:
            bucket = 'OTM'
        
        cases.append(dict(S0=S0, K=K, r=r, q=q, sigma=sigma, T=T, moneyness=(K / S0), bucket=bucket))

    return cases

# Main execution
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    ts = time.strftime('%Y%m%d-%H%M%S')
    outdir = os.path.join('./results', f'final_with_delta_loss_{ts}')
    ensure_dir(outdir)
    
    cases = sample_cases_aligned(
        S_range=(1.0, 4.0),
        r_lim=(0.02, 0.08),
        sigma_lim=(0.15, 0.45),
        K_abs=2.0,
        T_fixed=1.0,
        n_cases=30,
        atm_range=0.1,
    )
    
    N_STEPS = 160
    DEVICE = 'cpu'
    LR = 8e-4
    EPOCHS = 2000
    BATCH = 384
    
    rows = []
    for idx, cfg in enumerate(cases, 1):
        print(f"\n=== Option {idx}/{len(cases)} ({cfg['bucket']}) ===")
        print(json.dumps({k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in cfg.items()}, indent=2))
        
        model = AmericanCallRBSDE(
            S0=cfg['S0'], K=cfg['K'], r=cfg['r'], q=cfg['q'], sigma=cfg['sigma'], 
            T=cfg['T'], n_steps=N_STEPS, device=DEVICE, use_antithetic=True
        )
        
        model.train(
            n_epochs=EPOCHS, batch_size=BATCH, lr=LR, 
            verbose_every=250, patience=5000
        )
        
        am_abs_old, am_norm_old, term_mse, avg_push = evaluate_model(model)
        pred = model.predict_t0()
        am_abs = pred['price']
        am_norm = pred['price_norm']
        dn_delta = pred['delta']
        bs_abs = pred['bs_price']
        bs_delta = pred['bs_delta']
        eu_norm = bs_abs / cfg['S0']
        premium_norm = am_norm - eu_norm

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
        
    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, 'results_all.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved all results to: {csv_path}")

    plot_and_save(df, outdir)
    print(f"Figures saved to: {outdir}")

    stats_both = evaluate_vs_bs_both(df, atm_range=0.1)
    nn = stats_both['non_normalized']['price']
    nd = stats_both['non_normalized']['delta']
    nm = stats_both['normalized']['price']

    print("\n" + "="*60)
    print(" Evaluation Results (Non-normalized: absolute price)")
    print("="*60)
    print(f" MAE: {nn['MAE']:.6f}")
    print(f" MSE: {nn['MSE']:.6f}")
    print(f" RMSE: {nn['RMSE']:.6f}")
    print(f" Correlation: {nn['Correlation']:.4f}")
    print(f" RelErr vs BS (all): {nn['RelErr_vs_BS_pct']:.2f}%")
    print(f" RelErr vs BS (>0.1): {nn['RelErr_vs_BS_pct_filtered']:.2f}% [recommended]")
    print(f" RelErr vs S0: {nn['RelErr_vs_S0_pct']:.2f}%")
    print(f" sMAPE: {nn['sMAPE_pct']:.2f}%")
    print("-"*30)
    print(f" Delta MAE: {nd['MAE']:.6f}")
    print(f" Delta MSE: {nd['MSE']:.6f}")
    print(f" Delta RMSE: {nd['RMSE']:.6f}")
    print(f" Delta Correlation: {nd['Correlation']:.4f}")

    print("\n" + "="*60)
    print(" Evaluation Results (Normalized: price / S0)")
    print("="*60)
    print(f" MAE_norm: {nm['MAE']:.6f}")
    print(f" MSE_norm: {nm['MSE']:.6f}")
    print(f" RMSE_norm: {nm['RMSE']:.6f}")
    print(f" Corr_norm: {nm['Correlation']:.4f}")
    print(f" RelErr_norm vs BS (all): {nm['RelErr_vs_BS_pct']:.2f}%")
    print(f" RelErr_norm vs BS (>0.1): {nm['RelErr_vs_BS_pct_filtered']:.2f}% [recommended]")
    print(f" RelErr_norm vs S0: {nm['RelErr_vs_S0_pct']:.2f}% [note: normalized price, for reference]")
    print(f" sMAPE_norm: {nm['sMAPE_pct']:.2f}%")

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
        print(f" {b}: count={m['count']}, MAE={m['MAE']:.6f}, MSE={m['MSE']:.6f}, RMSE={m['RMSE']:.6f}")
    
    print("\nPer-bucket (normalized):")
    for b, m in pb['normalized'].items():
        print(f" {b}: count={m['count']}, MAE={m['MAE']:.6f}, MSE={m['MSE']:.6f}, RMSE={m['RMSE']:.6f}")

if __name__ == '__main__':
    main()