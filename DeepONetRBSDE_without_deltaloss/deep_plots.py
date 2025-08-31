import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import jax.numpy as jnp
from rbsde_solver import RBSDESolver
from deep_models import DeepONetArchitecture

# Plotting and Visualization

def save_training_plots(training_info):
    # Save training loss curve
    losses = training_info['losses']
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=150)
    plt.close()

    # Save loss components (log scale)
    if training_info.get('loss_components'):
        comps = training_info['loss_components']
        iters = np.arange(len(comps))
        stride = max(1, len(comps) // 200)
        plt.figure(figsize=(9, 6))
        plt.plot(iters[::stride], [c['terminal'] for c in comps][::stride], label='Terminal')
        plt.plot(iters[::stride], [c['rbsde_step'] for c in comps][::stride], label='RBSDE Step')
        plt.plot(iters[::stride], [c['rbsde_global'] for c in comps][::stride], label='RBSDE Global')
        plt.plot(iters[::stride], [c['obstacle_metric'] for c in comps][::stride], label='Obstacle')
        plt.plot(iters[::stride], [c['complementarity'] for c in comps][::stride], label='Complementarity')
        plt.plot(iters[::stride], [c['monotonicity'] for c in comps][::stride], label='Monotonicity')
        plt.plot(iters[::stride], [c['initial_K'] for c in comps][::stride], label='Initial K')
        plt.plot(iters[::stride], [c['K_terminal'] for c in comps][::stride], label='K Terminal')
        plt.title('Loss Components Evolution')
        plt.xlabel('Iteration')
        plt.ylabel('Loss (log)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('loss_components.png', dpi=150)
        plt.close()


#Generates and saves 3D surface plots and 2D heatmaps of the American premium.
def save_premium_surfaces(solver: RBSDESolver, networks: DeepONetArchitecture,
                          S0_range, sigma_range, r_range,
                          n_S0, n_sigma, n_r, r_fixed, sigma_fixed):
   
    # (1) r fixed: premium = f(S0, sigma)
    S0_grid = jnp.linspace(S0_range[0], S0_range[1], n_S0)
    sigma_grid = jnp.linspace(sigma_range[0], sigma_range[1], n_sigma)
    S0g1, sigmag1, prem1 = solver.premium_grid_r_fixed(networks, S0_grid, sigma_grid, r_fixed)

    fig = plt.figure(figsize=(8.8, 6.8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(np.asarray(S0g1), np.asarray(sigmag1), np.asarray(prem1),
                           linewidth=0.0, antialiased=True, rstride=1, cstride=1)
    ax.contour(np.asarray(S0g1), np.asarray(sigmag1), np.asarray(prem1), offset=np.min(prem1), levels=12)
    ax.view_init(elev=28, azim=135)
    ax.set_box_aspect((1.4, 1.0, 0.6))
    ax.set_title(f'American Premium Surface: r fixed = {r_fixed:.3f}')
    ax.set_xlabel('Spot S0')
    ax.set_ylabel('Sigma')
    ax.set_zlabel('Premium (DN - BS)')
    fig.colorbar(surf, ax=ax, shrink=0.72, pad=0.08)
    plt.tight_layout()
    plt.savefig('premium_surface_r_fixed.png', dpi=200)
    plt.close()

    # (2) sigma fixed: premium = f(S0, r)
    r_grid = jnp.linspace(r_range[0], r_range[1], n_r)
    S0g2, rg2, prem2 = solver.premium_grid_sigma_fixed(networks, S0_grid, r_grid, sigma_fixed)

    fig = plt.figure(figsize=(8.8, 6.8))
    ax = fig.add_subplot(111, projection='3d')
    surf2 = ax.plot_surface(np.asarray(S0g2), np.asarray(rg2), np.asarray(prem2),
                            linewidth=0.0, antialiased=True, rstride=1, cstride=1)
    ax.contour(np.asarray(S0g2), np.asarray(rg2), np.asarray(prem2), offset=np.min(prem2), levels=12)
    ax.view_init(elev=28, azim=135)
    ax.set_box_aspect((1.4, 1.0, 0.6))
    ax.set_title(f'American Premium Surface: sigma fixed = {sigma_fixed:.3f}')
    ax.set_xlabel('Spot S0')
    ax.set_ylabel('Rate r')
    ax.set_zlabel('Premium (DN - BS)')
    fig.colorbar(surf2, ax=ax, shrink=0.72, pad=0.08)
    plt.tight_layout()
    plt.savefig('premium_surface_sigma_fixed.png', dpi=200)
    plt.close()

    # 2D Heatmaps
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    cs = ax.contourf(np.asarray(S0g1), np.asarray(sigmag1), np.asarray(prem1), levels=18)
    fig.colorbar(cs, ax=ax, shrink=0.8, pad=0.02)
    ax.set_title(f'Premium Heatmap (r={r_fixed:.3f})')
    ax.set_xlabel('Spot S0')
    ax.set_ylabel('Sigma')
    plt.tight_layout()
    plt.savefig('premium_heatmap_r_fixed.png', dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    cs2 = ax.contourf(np.asarray(S0g2), np.asarray(rg2), np.asarray(prem2), levels=18)
    fig.colorbar(cs2, ax=ax, shrink=0.8, pad=0.02)
    ax.set_title(f'Premium Heatmap (sigma={sigma_fixed:.3f})')
    ax.set_xlabel('Spot S0')
    ax.set_ylabel('Rate r')
    plt.tight_layout()
    plt.savefig('premium_heatmap_sigma_fixed.png', dpi=180)
    plt.close()


def _nice_identity_line(ax, x, label='y = x'):
    lo = float(np.min(x))
    hi = float(np.max(x))
    ax.plot([lo, hi], [lo, hi], linewidth=1.2, linestyle='--', label=label)


def _bin_by_sigma(sigma, n_bins=4):

    qs = np.quantile(np.asarray(sigma), np.linspace(0, 1, n_bins + 1))
    bins = []
    for i in range(n_bins):
        lo, hi = qs[i], qs[i+1]
        s = np.asarray(sigma)
        mask = (s >= lo) & (s <= hi if i == n_bins - 1 else s < hi)
        bins.append((mask, f'σ∈[{lo:.3f}, {hi:.3f}]'))
    return bins


def save_validation_plots(results):
    dn = np.asarray(results['deeponet_prices'])
    bs = np.asarray(results['bs_prices'])
    dn_norm = np.asarray(results['deeponet_prices_normalized'])
    bs_norm = np.asarray(results['bs_prices_normalized'])
    S0 = np.asarray(results['test_data']['S0_values'])
    rvals = np.asarray(results['test_data']['r_values'])
    sigmas = np.asarray(results['test_data']['sigma_values'])

    resid = dn - bs
    resid_norm = dn_norm - bs_norm
    safe_bs = np.maximum(bs, 1e-6)
    rel_err_bs = (np.abs(resid) / safe_bs) * 100.0

    # Price Parity (non-normalized)
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.scatter(bs, dn, s=12, alpha=0.6)
    _nice_identity_line(ax, bs)
    ax.set_xlabel('BS Price')
    ax.set_ylabel('DeepONet Price')
    ax.set_title('Price Parity (DN vs BS)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('price_parity.png', dpi=170)
    plt.close()

    # Price Parity (normalized)
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.scatter(bs_norm, dn_norm, s=12, alpha=0.6)
    _nice_identity_line(ax, bs_norm)
    ax.set_xlabel('BS Price / S0')
    ax.set_ylabel('DeepONet Price / S0')
    ax.set_title('Normalized Price Parity (DN/S0 vs BS/S0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('price_parity_normalized.png', dpi=170)
    plt.close()

    # Residual Histograms (non-normalized and normalized)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(resid, bins=40, alpha=0.9)
    ax.set_xlabel('Residual (DN - BS)')
    ax.set_ylabel('Frequency')
    ax.set_title('Price Residual Histogram')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('price_residual_hist.png', dpi=170)
    plt.close()

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(resid_norm, bins=40, alpha=0.9)
    ax.set_xlabel('Residual (DN/S0 - BS/S0)')
    ax.set_ylabel('Frequency')
    ax.set_title('Normalized Price Residual Histogram')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('price_residual_hist_normalized.png', dpi=170)
    plt.close()

    # Relative error vs S0, sigma, and r
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(S0, rel_err_bs, s=10, alpha=0.6)
    ax.set_xlabel('S0')
    ax.set_ylabel('Rel. Error vs BS (%)')
    ax.set_title('Relative Price Error vs S0')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rel_error_vs_S0.png', dpi=170)
    plt.close()

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(sigmas, rel_err_bs, s=10, alpha=0.6)
    ax.set_xlabel('Sigma')
    ax.set_ylabel('Rel. Error vs BS (%)')
    ax.set_title('Relative Price Error vs Sigma')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rel_error_vs_sigma.png', dpi=170)
    plt.close()

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(rvals, rel_err_bs, s=10, alpha=0.6)
    ax.set_xlabel('Rate r')
    ax.set_ylabel('Rel. Error vs BS (%)')
    ax.set_title('Relative Price Error vs r')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rel_error_vs_r.png', dpi=170)
    plt.close()


def save_delta_plots(results, solver: RBSDESolver, networks: DeepONetArchitecture, r_fixed=0.05):
    
    d_dn = np.asarray(results['deeponet_deltas'])
    d_bs = np.asarray(results['bs_deltas'])
    Z_dn = np.asarray(results['deeponet_Z0_from_delta'])
    Z_bs = np.asarray(results['bs_Z0'])
    S0 = np.asarray(results['test_data']['S0_values'])
    sigmas = np.asarray(results['test_data']['sigma_values'])
    d_err = d_dn - d_bs

    # Delta Parity
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.scatter(d_bs, d_dn, s=12, alpha=0.6)
    _nice_identity_line(ax, d_bs)
    ax.set_xlabel('BS Delta')
    ax.set_ylabel('DeepONet Delta')
    ax.set_title('Delta Parity (DN vs BS)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('delta_parity.png', dpi=170)
    plt.close()

    # Z Parity
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.scatter(Z_bs, Z_dn, s=12, alpha=0.6)
    _nice_identity_line(ax, Z_bs)
    ax.set_xlabel('BS Z = Δ·σ·S')
    ax.set_ylabel('DN Z (from Δ)')
    ax.set_title('Z Parity (DN vs BS)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Z_parity.png', dpi=170)
    plt.close()

    # Delta Error Histogram
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(d_err, bins=40, alpha=0.9)
    ax.set_xlabel('Delta Error (DN - BS)')
    ax.set_ylabel('Frequency')
    ax.set_title('Delta Error Histogram')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('delta_error_hist.png', dpi=170)
    plt.close()

    # Delta vs S0 (grouped by sigma)
    bins = _bin_by_sigma(sigmas, n_bins=4)
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for mask, lab in bins:
        if np.sum(mask) < 5:
            continue
        idx = np.argsort(S0[mask])
        s_sorted = S0[mask][idx]
        d_sorted = d_dn[mask][idx]
        win = max(5, len(s_sorted) // 30)
        kernel = np.ones(win) / win
        s_smooth = s_sorted[win - 1:]
        d_smooth = np.convolve(d_sorted, kernel, mode='valid')
        ax.plot(s_smooth, d_smooth, label=lab)
    ax.set_xlabel('S0')
    ax.set_ylabel('DeepONet Delta (smoothed)')
    ax.set_title('Delta vs S0 (grouped by sigma)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig('delta_vs_S0_sigma_groups.png', dpi=170)
    plt.close()

    # Delta Heatmap (fixed r)
    S0_grid = jnp.linspace(solver.config.S_lim[0], solver.config.S_lim[1], 80)
    sigma_grid = jnp.linspace(solver.config.sigma_lim[0], solver.config.sigma_lim[1], 80)
    S0g, sigmag = jnp.meshgrid(S0_grid, sigma_grid, indexing='xy')
    S0_flat, sigma_flat = S0g.reshape(-1), sigmag.reshape(-1)
    r_flat = jnp.full_like(S0_flat, r_fixed)

    K_hat = solver.config.K / S0_flat
    t0, s_hat0 = jnp.zeros_like(S0_flat), jnp.ones_like(S0_flat)
    trunk_input = jnp.stack([t0, s_hat0, sigma_flat, r_flat], axis=1)
    sensor_points_hat = jnp.linspace(solver.config.shat_min, solver.config.shat_max, solver.config.n_sensors)
    branch_per_path = jnp.maximum(sensor_points_hat[None, :] - K_hat[:, None], 0.0)
    branch_input = branch_per_path[:, :, None]

    Z0 = solver.compute_Z(branch_input, trunk_input, networks)
    denom = jnp.maximum(sigma_flat * jnp.maximum(S0_flat, 1e-6), 1e-6)
    delta0 = (Z0 / denom).reshape(S0g.shape)

    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    cs = ax.contourf(np.asarray(S0g), np.asarray(sigmag), np.asarray(delta0), levels=18)
    fig.colorbar(cs, ax=ax, shrink=0.8, pad=0.02)
    ax.set_title(f'DeepONet Delta Heatmap (r={r_fixed:.3f})')
    ax.set_xlabel('Spot S0')
    ax.set_ylabel('Sigma')
    plt.tight_layout()
    plt.savefig('delta_heatmap_r_fixed.png', dpi=180)
    plt.close()