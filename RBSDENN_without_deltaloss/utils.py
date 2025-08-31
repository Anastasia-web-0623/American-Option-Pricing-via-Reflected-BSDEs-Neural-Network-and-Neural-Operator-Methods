import os
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
import math
import torch
from typing import Tuple, Dict

#Black-Scholes model for european call option
def black_scholes_call_price(S, K, T, r, sigma, q=0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return max(call_price, 0.0)

def black_scholes_delta(S, K, T, r, sigma, q=0.0) -> float:
    if T <= 1e-12:
        return float(1.0 if S >= K else 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    Nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    return float(np.exp(-q * T) * Nd1)
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# Sampling function to generate diverse option cases,ATM OTM and ITM to get values
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

# Core function to compute evaluation metrics
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


# Comprehensive evaluation function,including normalized and non-normalized
def evaluate_vs_bs_both(results_df: pd.DataFrame, atm_range: float = 0.1) -> dict:

    df = results_df.copy()
    S0 = df['S0'].to_numpy()
    K = df['K'].to_numpy()

    # price metrics
    abs_metrics = _core_price_metrics(df['Am_Price'].to_numpy(), df['Eu_Price'].to_numpy(), S0)
    norm_metrics = _core_price_metrics(df['Am_Price_norm'].to_numpy(), df['Eu_Price_norm'].to_numpy(), S0)

    # delta metrics (no normalization)
    dd = df['Am_Delta'].to_numpy()
    bd = df['BS_Delta'].to_numpy()
    delta_mae = float(np.abs(dd - bd).mean())
    delta_mse = float(((dd - bd) ** 2).mean())
    delta_rmse = float(np.sqrt(delta_mse))
    delta_corr = float(np.corrcoef(dd, bd)[0, 1]) if dd.size > 1 else np.nan
    delta_metrics = dict(MAE=delta_mae, MSE=delta_mse, RMSE=delta_rmse, Correlation=delta_corr)

    # groups with ITM ATM and OTM
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
