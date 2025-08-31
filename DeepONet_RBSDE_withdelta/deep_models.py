import jax
import jax.numpy as jnp
import jax.scipy
import equinox as eqx
from typing import NamedTuple

#european call pricer, the Black-Scholes price model and delta value

class EuropeanCallPricer:
    @staticmethod
    def price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 1e-8:
            return jnp.maximum(S - K, 0.0)
        d1 = (jnp.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
        d2 = d1 - sigma * jnp.sqrt(T)
        N_d1 = 0.5 * (1.0 + jax.scipy.special.erf(d1 / jnp.sqrt(2.0)))
        N_d2 = 0.5 * (1.0 + jax.scipy.special.erf(d2 / jnp.sqrt(2.0)))
        return S * N_d1 - K * jnp.exp(-r * T) * N_d2
    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 1e-8:
            return jnp.where(S >= K, 1.0, 0.0)
        d1 = (jnp.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
        return 0.5 * (1.0 + jax.scipy.special.erf(d1 / jnp.sqrt(2.0)))

"""
Each DeepONet consists of a branch network, a trunk network, and a bias network.
    - Y_branch/Y_trunk/Y_bias: For the value function Y.
    - Z_branch/Z_trunk/Z_bias: For the Delta-like process Z.
    - K_branch/K_trunk/K_bias: For the non-decreasing process K.
"""
class DeepONetArchitecture(NamedTuple):
    Y_branch: eqx.Module
    Y_trunk: eqx.Module
    Y_bias: eqx.Module
    Z_branch: eqx.Module
    Z_trunk: eqx.Module
    Z_bias: eqx.Module
    K_branch: eqx.Module
    K_trunk: eqx.Module
    K_bias: eqx.Module
    
    def build_mlp_networks(config, key: jax.random.PRNGKey) -> 'DeepONetArchitecture':
      
        keys = jax.random.split(key, 9)
        act = jax.nn.tanh

        # DeepONet for the value process Y
        Y_branch = eqx.nn.MLP(1, config.latent_dim, config.branch_width, config.depth, act, key=keys[0])
        Y_trunk  = eqx.nn.MLP(4, config.latent_dim, config.trunk_width,  config.depth, act, key=keys[1])
        Y_bias   = eqx.nn.MLP(4, 1, 32, 2, jax.nn.identity, key=keys[2])

        # DeepONet for the Z process
        Z_branch = eqx.nn.MLP(1, config.latent_dim, config.branch_width, config.depth, act, key=keys[3])
        Z_trunk  = eqx.nn.MLP(4, config.latent_dim, config.trunk_width,  config.depth, act, key=keys[4])
        Z_bias   = eqx.nn.MLP(4, 1, 32, 2, jax.nn.identity, key=keys[5])

        # DeepONet for the K process
        K_branch = eqx.nn.MLP(1, config.latent_dim, config.branch_width, config.depth, act, key=keys[6])
        K_trunk  = eqx.nn.MLP(4, config.latent_dim, config.trunk_width,  config.depth, act, key=keys[7])
        K_bias   = eqx.nn.MLP(4, 1, 32, 2, jax.nn.identity, key=keys[8])

        return DeepONetArchitecture(Y_branch, Y_trunk, Y_bias, Z_branch, Z_trunk, Z_bias, K_branch, K_trunk, K_bias)
    

