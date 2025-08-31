import os
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def plot_and_save(results_df: pd.DataFrame, outdir: str):
    # 1) Premium vs Bucket (norm)
    g = results_df.groupby('bucket')['Am_Premium_norm'].mean().reset_index()
    plt.figure()
    plt.plot(g['bucket'], g['Am_Premium_norm'], marker='o')
    plt.grid(True, ls='--', lw=0.5)
    plt.title('Early Exercise Premium (norm) by Bucket')
    plt.xlabel('Bucket')
    plt.ylabel('Premium (norm)')
    plt.savefig(os.path.join(outdir, 'premium_by_bucket.png'), bbox_inches='tight')
    plt.close()

    # 2) Terminal MSE vs Bucket
    g2 = results_df.groupby('bucket')['Terminal_MSE_norm'].mean().reset_index()
    plt.figure()
    plt.plot(g2['bucket'], g2['Terminal_MSE_norm'], marker='o')
    plt.grid(True, ls='--', lw=0.5)
    plt.title('Terminal MSE (norm) by Bucket')
    plt.xlabel('Bucket')
    plt.ylabel('Terminal MSE (norm)')
    plt.savefig(os.path.join(outdir, 'mse_by_bucket.png'), bbox_inches='tight')
    plt.close()

    # 3) Avg Push vs Bucket
    g3 = results_df.groupby('bucket')['Avg_Total_Push_norm'].mean().reset_index()
    plt.figure()
    plt.plot(g3['bucket'], g3['Avg_Total_Push_norm'], marker='o')
    plt.grid(True, ls='--', lw=0.5)
    plt.title('Avg Total Push (norm) by Bucket')
    plt.xlabel('Bucket')
    plt.ylabel('Avg Push (norm)')
    plt.savefig(os.path.join(outdir, 'push_by_bucket.png'), bbox_inches='tight')
    plt.close()

    # 4) y0_pred vs y0_target (norm)
    plt.figure()
    plt.scatter(results_df['y0_pred_norm'], results_df['y0_target_norm'], alpha=0.7)
    lo = min(results_df['y0_pred_norm'].min(), results_df['y0_target_norm'].min())
    hi = max(results_df['y0_pred_norm'].max(), results_df['y0_target_norm'].max())
    plt.plot([lo, hi], [lo, hi], ls='--')
    plt.grid(True, ls='--', lw=0.5)
    plt.title('y0_pred vs y0_target (norm)')
    plt.xlabel('y0_pred (norm)')
    plt.ylabel('y0_target (norm)')
    plt.savefig(os.path.join(outdir, 'y0_scatter.png'), bbox_inches='tight')
    plt.close()
    
    # 5) Absolute Price vs BS
    plt.figure()
    plt.scatter(results_df['Eu_Price'], results_df['Am_Price'], alpha=0.7)
    lo = float(min(results_df['Eu_Price'].min(), results_df['Am_Price'].min()))
    hi = float(max(results_df['Eu_Price'].max(), results_df['Am_Price'].max()))
    plt.plot([lo, hi], [lo, hi], ls='--')
    plt.grid(True, ls='--', lw=0.5)
    plt.title('American Price vs Black–Scholes (absolute)')
    plt.xlabel('BS European Price')
    plt.ylabel('Deep RBSDE American Price')
    plt.savefig(os.path.join(outdir, 'price_vs_bs.png'), bbox_inches='tight')
    plt.close()

    # 6) Delta vs BS
    if 'Am_Delta' in results_df.columns and 'BS_Delta' in results_df.columns:
        plt.figure()
        plt.scatter(results_df['BS_Delta'], results_df['Am_Delta'], alpha=0.7)
        plt.plot([0,1], [0,1], ls='--')
        plt.grid(True, ls='--', lw=0.5)
        plt.title('Delta (American) vs Delta (BS)')
        plt.xlabel('BS Delta')
        plt.ylabel('American Delta (Z/(σ))')
        plt.savefig(os.path.join(outdir, 'delta_vs_bs.png'), bbox_inches='tight')
        plt.close()

    # 7) Premium (Absolute) by bucket
    results_df['Am_Premium'] = results_df['Am_Price'] - results_df['Eu_Price']
    g4 = results_df.groupby('bucket')['Am_Premium'].mean().reset_index()
    plt.figure()
    plt.plot(g4['bucket'], g4['Am_Premium'], marker='o')
    plt.grid(True, ls='--', lw=0.5)
    plt.title('Early Exercise Premium (Absolute) by Bucket')
    plt.xlabel('Bucket')
    plt.ylabel('Premium (Absolute)')
    plt.savefig(os.path.join(outdir, 'premium_by_bucket_abs.png'), bbox_inches='tight')
    plt.close()
    
    # 8) Normalized Price vs BS/S0
    plt.figure()
    plt.scatter(results_df['Eu_Price_norm'], results_df['Am_Price_norm'], alpha=0.7)
    lo = float(min(results_df['Eu_Price_norm'].min(), results_df['Am_Price_norm'].min()))
    hi = float(max(results_df['Eu_Price_norm'].max(), results_df['Am_Price_norm'].max()))
    plt.plot([lo, hi], [lo, hi], ls='--')
    plt.grid(True, ls='--', lw=0.5)
    plt.title('American Price/S0 vs BS Price/S0 (normalized)')
    plt.xlabel('BS Price / S0')
    plt.ylabel('American Price / S0')
    plt.savefig(os.path.join(outdir, 'price_parity_normalized.png'), bbox_inches='tight')
    plt.close()
    
    # 9) Residual hist (abs & norm)
    resid_abs = results_df['Am_Price'] - results_df['Eu_Price']
    resid_norm = results_df['Am_Price_norm'] - results_df['Eu_Price_norm']
    plt.figure()
    plt.hist(resid_abs, bins=30, alpha=0.9)
    plt.grid(True, ls='--', lw=0.5)
    plt.xlabel('Residual (DN - BS)')
    plt.ylabel('Frequency')
    plt.title('Price Residual Histogram (absolute)')
    plt.savefig(os.path.join(outdir, 'price_residual_hist_abs.png'), bbox_inches='tight')
    plt.close()
    
    plt.figure()
    plt.hist(resid_norm, bins=30, alpha=0.9)
    plt.grid(True, ls='--', lw=0.5)
    plt.xlabel('Residual (DN/S0 - BS/S0)')
    plt.ylabel('Frequency')
    plt.title('Price Residual Histogram (normalized)')
    plt.savefig(os.path.join(outdir, 'price_residual_hist_norm.png'), bbox_inches='tight')
    plt.close()