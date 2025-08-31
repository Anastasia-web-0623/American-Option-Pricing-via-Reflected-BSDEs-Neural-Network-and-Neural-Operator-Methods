from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#  Black–Scholes (no dividends)
def _std_norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_price(
    *,
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S0 * _std_norm_cdf(d1) - K * math.exp(-r * T) * _std_norm_cdf(d2)
    return K * math.exp(-r * T) * _std_norm_cdf(-d2) - S0 * _std_norm_cdf(-d1)


#  Bases (Laguerre / Hermite / Polynomial)

def laguerre_polynomial(x: np.ndarray, n: int) -> np.ndarray:  # L_n(x)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 1.0 - x
    L0, L1 = np.ones_like(x), 1.0 - x
    for k in range(2, n + 1):
        L2 = ((2 * k - 1 - x) * L1 - (k - 1) * L0) / k
        L0, L1 = L1, L2
    return L1

def hermite_polynomial(x: np.ndarray, n: int) -> np.ndarray:  # He_n(x) (probabilists')
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x
    H0, H1 = np.ones_like(x), x
    for k in range(2, n + 1):
        H2 = x * H1 - (k - 1) * H0
        H0, H1 = H1, H2
    return H1

BasisFunc = Callable[[np.ndarray, int], np.ndarray]
_BASIS_REGISTRY: Dict[str, BasisFunc] = {
    "laguerre": laguerre_polynomial,
    "hermite": hermite_polynomial,
    "polynomial": lambda x, n: x ** n,
}

#  Longstaff–Schwartz Monte Carlo engine

@dataclass
class AmericanOptionLSM:
    S0: float
    K: float
    T: float
    r: float
    sigma: float
    n_paths: int = 20_000
    n_steps: int = 50
    basis: str = "laguerre"
    degree: int = 4
    option_type: str = "call"
    seed: int | None = None

    # internals – recorded during pricing
    regression_mse: List[float] = field(default_factory=list, init=False)

    def _intrinsic(self, S: np.ndarray) -> np.ndarray:
        if self.option_type == "call":
            return np.maximum(S - self.K, 0.0)
        return np.maximum(self.K - S, 0.0)

    def _paths(self) -> np.ndarray:
        if self.seed is not None:
            np.random.seed(self.seed)
        dt = self.T / self.n_steps
        paths = np.zeros((self.n_paths, self.n_steps + 1))
        paths[:, 0] = self.S0
        dW = np.random.normal(0.0, math.sqrt(dt), size=(self.n_paths, self.n_steps))
        drift = (self.r - 0.5 * self.sigma**2) * dt
        vol = self.sigma * dW
        for t in range(1, self.n_steps + 1):
            paths[:, t] = paths[:, t - 1] * np.exp(drift + vol[:, t - 1])
        return paths

    def _basis(self, S: np.ndarray) -> np.ndarray:
        if self.degree <= 0:
            raise ValueError("degree must be >= 1")
        S_norm = S / self.K  # normalize for conditioning
        basis_fn = _BASIS_REGISTRY[self.basis]
        return np.column_stack([basis_fn(S_norm, k) for k in range(self.degree)])
    def price(self, debug: bool = False) -> float:
        dt = self.T / self.n_steps
        disc = math.exp(-self.r * dt)
        paths = self._paths()
        cashflows = self._intrinsic(paths[:, -1])

        self.regression_mse.clear()
        for t in range(self.n_steps - 1, 0, -1):
            cashflows *= disc
            S_t = paths[:, t]                 # (bug fix) ensure S_t defined
            h_t = self._intrinsic(S_t)
            itm = h_t > 0.0
            if itm.sum() < self.degree:
                continue  # avoid ill-posed regression
            X = self._basis(S_t[itm])
            Y = cashflows[itm]
            coeff, *_ = np.linalg.lstsq(X, Y, rcond=None)
            continuation = X @ coeff
            self.regression_mse.append(float(np.mean((Y - continuation) ** 2)))
            exercise = h_t[itm] > continuation
            cashflows[itm] = np.where(exercise, h_t[itm], Y)
            if debug and t % 10 == 0:
                print(f"t={t:02d} | ITM={itm.sum():5d} | exc={exercise.sum():5d}")
        return float(cashflows.mean())

#  Sampling & Experiment Orchestration
def sample_option_points(
    *,
    K: float = 2.0,
    T: float = 1.0,
    S_lim: Tuple[float, float] = (1.0, 4.0),
    sigma_lim: Tuple[float, float] = (0.15, 0.45),
    r_lim: Tuple[float, float] = (0.02, 0.08),
    option_type: str = "call",
    num_samples: int = 50,
    sample_seed: int = 2025,
) -> List[Dict]:
    """Uniformly sample parameter points (S0, sigma, r) within given ranges."""
    rng = np.random.default_rng(sample_seed)
    samples: List[Dict] = []
    for _ in range(num_samples):
        S0 = float(rng.uniform(S_lim[0], S_lim[1]))
        sigma = float(rng.uniform(sigma_lim[0], sigma_lim[1]))
        r = float(rng.uniform(r_lim[0], r_lim[1]))
        samples.append(dict(S0=S0, K=K, T=T, r=r, sigma=sigma, option_type=option_type))
    return samples

def run_all_combos(
    samples: List[Dict],
    basis_list: Sequence[str],
    degree_list: Sequence[int],
    *,
    n_runs: int = 10,
    n_paths: int = 20_000,
    n_steps: int = 50,
) -> pd.DataFrame:
    """Run LSM for all (sample × basis × degree × seed) and collect records."""
    records: List[Dict] = []
    for sid, params in enumerate(samples):
        target = bs_price(**params)  # for non-div call, American == European
        for basis in basis_list:
            for degree in degree_list:
                for run in range(n_runs):
                    model = AmericanOptionLSM(
                        **params,
                        n_paths=n_paths,
                        n_steps=n_steps,
                        basis=basis,
                        degree=degree,
                        seed=run,
                    )
                    price = model.price()
                    err = price - target
                    mean_reg_mse = float(np.mean(model.regression_mse)) if model.regression_mse else float("nan")
                    records.append(
                        dict(
                            sample_id=sid,
                            run=run,
                            basis=basis,
                            degree=degree,
                            S0=params["S0"],
                            K=params["K"],
                            T=params["T"],
                            r=params["r"],
                            sigma=params["sigma"],
                            moneyness=params["S0"] / params["K"],
                            price=price,
                            target=target,
                            abs_error=abs(err),
                            sq_error=err**2,
                            mean_reg_mse=mean_reg_mse,  # regression MSE (fit quality)
                        )
                    )
    return pd.DataFrame.from_records(records)

def summarize_combos(
    df: pd.DataFrame,
    *,
    w_mae: float = 0.5,
    w_rmse: float = 0.5,
) -> pd.DataFrame:
    """Aggregate to per-(basis, degree) summary with MAE, RMSE and a joint score."""
    summary = (
        df.groupby(["basis", "degree"], sort=False)
        .agg(
            mae=("abs_error", "mean"),
            rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
            p95_abs_error=("abs_error", lambda x: float(np.percentile(x, 95))),
            mean_reg_mse=("mean_reg_mse", "mean"),
        )
        .reset_index()
    )

    # min–max normalization for joint score
    eps = 1e-12
    mae_min, mae_max = summary["mae"].min(), summary["mae"].max()
    rmse_min, rmse_max = summary["rmse"].min(), summary["rmse"].max()
    summary["mae_norm"] = (summary["mae"] - mae_min) / (mae_max - mae_min + eps)
    summary["rmse_norm"] = (summary["rmse"] - rmse_min) / (rmse_max - rmse_min + eps)
    summary["joint_score"] = w_mae * summary["mae_norm"] + w_rmse * summary["rmse_norm"]  # lower is better

    return summary.sort_values(["joint_score", "mae", "rmse"]).reset_index(drop=True)

# =============================================================================
#  Plotting helpers
# =============================================================================
def _ensure_outdir(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def plot_bar(values: pd.Series, labels: List[str], title: str, ylabel: str, outdir: pathlib.Path, filename: str) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, values.values)
    ax.set_title(title)
    ax.set_xlabel("Combination")
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    file = outdir / filename
    fig.savefig(file, dpi=150)
    plt.close(fig)
    return file

def plot_mae_rmse_scatter(summary: pd.DataFrame, title: str, outdir: pathlib.Path, filename: str) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(summary["mae"], summary["rmse"])
    for _, row in summary.iterrows():
        ax.text(row["mae"], row["rmse"], f"{row['basis'][0]}{int(row['degree'])}", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("MAE")
    ax.set_ylabel("RMSE")
    fig.tight_layout()
    file = outdir / filename
    fig.savefig(file, dpi=150)
    plt.close(fig)
    return file

def plot_top_combo_diagnostics(df: pd.DataFrame, top: pd.Series, outdir: pathlib.Path) -> Dict[str, pathlib.Path]:
    mask = (df["basis"] == top["basis"]) & (df["degree"] == top["degree"])
    d = df.loc[mask].copy()

    files: Dict[str, pathlib.Path] = {}

    # 1) |error| histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(d["abs_error"], bins=30)
    ax.set_title(f"Top {top['basis']}-{int(top['degree'])}: |Error| distribution")
    ax.set_xlabel("|Error|")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    files["hist_error"] = outdir / "top_combo_error_hist.png"
    fig.savefig(files["hist_error"], dpi=150)
    plt.close(fig)

    # 2) |error| vs regression MSE
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(d["mean_reg_mse"], d["abs_error"])
    ax.set_title(f"Top {top['basis']}-{int(top['degree'])}: |Error| vs Regression MSE")
    ax.set_xlabel("Mean regression MSE (per run)")
    ax.set_ylabel("|Error|")
    fig.tight_layout()
    files["scatter_error_vs_regmse"] = outdir / "top_combo_error_vs_regmse.png"
    fig.savefig(files["scatter_error_vs_regmse"], dpi=150)
    plt.close(fig)

    # 3) |error| vs moneyness
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(d["moneyness"], d["abs_error"])
    ax.set_title(f"Top {top['basis']}-{int(top['degree'])}: |Error| vs Moneyness S0/K")
    ax.set_xlabel("Moneyness (S0/K)")
    ax.set_ylabel("|Error|")
    fig.tight_layout()
    files["scatter_error_vs_moneyness"] = outdir / "top_combo_error_vs_moneyness.png"
    fig.savefig(files["scatter_error_vs_moneyness"], dpi=150)
    plt.close(fig)

    # 4) |error| vs sigma
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(d["sigma"], d["abs_error"])
    ax.set_title(f"Top {top['basis']}-{int(top['degree'])}: |Error| vs sigma")
    ax.set_xlabel("sigma")
    ax.set_ylabel("|Error|")
    fig.tight_layout()
    files["scatter_error_vs_sigma"] = outdir / "top_combo_error_vs_sigma.png"
    fig.savefig(files["scatter_error_vs_sigma"], dpi=150)
    plt.close(fig)

    # 5) |error| vs r
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(d["r"], d["abs_error"])
    ax.set_title(f"Top {top['basis']}-{int(top['degree'])}: |Error| vs r")
    ax.set_xlabel("r")
    ax.set_ylabel("|Error|")
    fig.tight_layout()
    files["scatter_error_vs_r"] = outdir / "top_combo_error_vs_r.png"
    fig.savefig(files["scatter_error_vs_r"], dpi=150)
    plt.close(fig)

    return files


#  End-to-end validation demo


def validate_lsm_over_samples(
    *,
    K: float = 2.0,
    T: float = 1.0,
    S_lim: Tuple[float, float] = (1.0, 4.0),
    sigma_lim: Tuple[float, float] = (0.15, 0.45),
    r_lim: Tuple[float, float] = (0.02, 0.08),
    option_type: str = "call",
    num_samples: int = 50,
    sample_seed: int = 2025,
    basis_list: Sequence[str] = ("laguerre", "polynomial", "hermite"),
    degree_list: Sequence[int] = (2, 3, 4, 5),
    n_runs: int = 10,
    n_paths: int = 20_000,
    n_steps: int = 50,
    w_mae: float = 0.5,
    w_rmse: float = 0.5,
    outdir: str | pathlib.Path = "lsm_validation_outputs",
) -> Dict[str, pathlib.Path | str]:
    outdir = _ensure_outdir(pathlib.Path(outdir))
        # 1) sample points
    samples = sample_option_points(
        K=K, T=T,
        S_lim=S_lim, sigma_lim=sigma_lim, r_lim=r_lim,
        option_type=option_type,
        num_samples=num_samples, sample_seed=sample_seed,
    )

    # 2) run all combos
    df = run_all_combos(
        samples, basis_list, degree_list,
        n_runs=n_runs, n_paths=n_paths, n_steps=n_steps,
    )

    # 3) summaries (MAE, RMSE, p95, mean_reg_mse, joint score)
    summary = summarize_combos(df, w_mae=w_mae, w_rmse=w_rmse)

    # 4) save CSVs
    all_csv = outdir / "lsm_all_records.csv"
    summary_csv = outdir / "lsm_combo_summary_mae_rmse_mse.csv"
    df.to_csv(all_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    # 5) plots
    labels = [f"{b}\n{d}" for b, d in zip(summary["basis"], summary["degree"])]
    files: Dict[str, pathlib.Path | str] = {
        "csv_all_records": str(all_csv),
        "csv_summary": str(summary_csv),
        "bar_mae": str(plot_bar(summary["mae"], labels, "MAE by Basis & Degree", "MAE", outdir, "bar_mae.png")),
        "bar_rmse": str(plot_bar(summary["rmse"], labels, "RMSE by Basis & Degree", "RMSE", outdir, "bar_rmse.png")),
        "bar_joint_score": str(plot_bar(summary["joint_score"], labels, "Joint Score (MAE & RMSE)", "Score (lower is better)", outdir, "bar_joint_score.png")),
        "scatter_mae_vs_rmse": str(plot_mae_rmse_scatter(summary, "MAE vs RMSE (lower-left better)", outdir, "scatter_mae_vs_rmse.png")),
    }

    # 6) diagnostics for the top combo
    top = summary.iloc[0]
    diag = plot_top_combo_diagnostics(df, top, outdir)
    files.update({k: str(v) for k, v in diag.items()})

    # 7) print a brief summary
    print("\nBest combination by joint score:")
    print(top[["basis", "degree", "mae", "rmse", "p95_abs_error", "mean_reg_mse", "joint_score"]])

    return files

# -----------------------------------------------------------------------------
# Script entry
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    #   K=2.0, T=1.0,
    #   S0 ∈ [1, 4], sigma ∈ [0.15, 0.45], r ∈ [0.02, 0.08]
    # 可按需增大 num_samples / n_runs / n_paths / n_steps 提升精度。
    validate_lsm_over_samples(
        K=2.0,
        T=1.0,
        S_lim=(1.0, 4.0),
        sigma_lim=(0.15, 0.45),
        r_lim=(0.02, 0.08),
        option_type="call",
        num_samples=50,
        sample_seed=2025,
        basis_list=("laguerre", "polynomial", "hermite"),
        degree_list=(2, 3, 4, 5),
        n_runs=10,
        n_paths=20_000,
        n_steps=50,
        w_mae=0.5,
        w_rmse=0.5,
        outdir="lsm_validation_outputs",
    )