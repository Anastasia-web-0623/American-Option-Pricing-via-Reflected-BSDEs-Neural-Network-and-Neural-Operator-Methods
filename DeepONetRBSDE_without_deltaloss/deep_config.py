import jax.numpy as jnp
from typing import Tuple,NamedTuple

class RBSDEConfig(NamedTuple):
    #the settings and hyperparameters for the model and training process and loss function weights
    K: float = 2.0  
    T: float = 1.0  
    S_lim: Tuple[float, float] = (1.0, 4.0)  
    sigma_lim: Tuple[float, float] = (0.15, 0.45)  
    r_lim: Tuple[float, float] = (0.02, 0.08) 
    # Network architecture parameters
    n_sensors: int = 25
    branch_width: int = 64
    trunk_width: int = 64
    latent_dim: int = 64
    depth: int = 4

    # Training parameters
    n_paths: int = 300  # Number of paths for generating training data
    n_steps: int = 50   # Number of time steps in each path
    batch_size: int = 128
    learning_rate: float = 1e-3
    n_iterations: int = 3000
    # RBSDE penalties / weights for the loss function
    obstacle_penalty: float = 172.093323
    complementarity_penalty: float = 107.612915
    smoothing_eps: float = 1e-2

    terminal_weight: float = 586.200562
    rbsde_step_weight: float = 65.726707
    rbsde_global_weight: float = 36.914509
    monotonicity_weight: float = 93.372665
    initial_K_weight: float = 32.112041
    k_terminal_weight: float = 0.671314
    z_box_weight: float = 4.376514  # Kept for backward compatibility, but unused

    # Weights for different option moneyness categories
    itm_weight: float = 2.697  
    atm_weight: float = 1.288 
    otm_weight: float = 0.107  
    atm_range: float = 0.1     
    # Branch sensor grid in normalized space (shat = S/S0)
    shat_min: float = 0.5
    shat_max: float = 2.5

#compute the mean relative error
def masked_rel_err(yhat: jnp.ndarray, y: jnp.ndarray, threshold: float = 0.1, eps: float = 1e-8) -> float:
    mask = jnp.abs(y) > threshold
    if not bool(jnp.any(mask)):
        return float("nan")
    num = jnp.abs(yhat[mask] - y[mask])
    den = jnp.clip(jnp.abs(y[mask]), eps, None)
    return float(jnp.mean(num / den))
#compute the symmertic mean absolute percentage error
def smape(yhat: jnp.ndarray, y: jnp.ndarray, eps: float = 1e-8) -> float:
    den = jnp.clip(jnp.abs(yhat) + jnp.abs(y), eps, None)
    return float(jnp.mean(2.0 * jnp.abs(yhat - y) / den))

#Calculates the pointwise relative error scaled by the initial spot price (S0).
def rel_err_vs_s0(yhat: jnp.ndarray, y: jnp.ndarray, S0: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    return jnp.abs(yhat - y) / jnp.clip(jnp.abs(S0), eps, None)
