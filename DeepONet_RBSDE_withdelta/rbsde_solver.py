import jax
import jax.numpy as jnp
import optax
import equinox as eqx

from deep_config import RBSDEConfig, masked_rel_err, smape, rel_err_vs_s0
from deep_models import EuropeanCallPricer, DeepONetArchitecture

#solve the RBSDE for american option pricing
class RBSDESolver:
    def __init__(self, config: RBSDEConfig):
        self.config = config
        self.euro = EuropeanCallPricer()

# Calculates the normalized payoff function.S_hat = S/S0, K_hat = K/S0. Payoff is (S-K)/S0.
    @staticmethod
    def payoff_hat(S_hat: jnp.ndarray, K_hat: jnp.ndarray) -> jnp.ndarray:
        
        return jnp.maximum(S_hat - K_hat, 0.0)

    def deeponet_forward(self, branch_input, trunk_input, branch_net, trunk_net, bias_net):
        # branch_input: (B, n_sensors, 1) ; trunk_input: (B, 4)
        B, n, _ = branch_input.shape
        branch_input_flat = branch_input.reshape(-1, 1)                    
        branch_out_flat = jax.vmap(branch_net)(branch_input_flat)          
        branch_out = branch_out_flat.reshape(B, n, -1).mean(axis=1)        
        trunk_out = jax.vmap(trunk_net)(trunk_input)                       
        bias = jax.vmap(bias_net)(trunk_input)                           
        out = (branch_out * trunk_out).sum(axis=1, keepdims=True) + bias  
        return out.squeeze(-1) 
        
#compute the normalized value of Y
    def compute_Yhat(self, branch_input, trunk_input, nets: DeepONetArchitecture):
        """Computes the normalized value function Y_hat."""
        return self.deeponet_forward(branch_input, trunk_input, nets.Y_branch, nets.Y_trunk, nets.Y_bias)
#compute the z value
    def compute_Z(self, branch_input, trunk_input, nets: DeepONetArchitecture):
        return self.deeponet_forward(branch_input, trunk_input, nets.Z_branch, nets.Z_trunk, nets.Z_bias)

    def compute_Khat(self, branch_input, trunk_input, nets: DeepONetArchitecture):
        """Computes the non-decreasing process K_hat, related to early exercise."""
        K_raw = self.deeponet_forward(branch_input, trunk_input, nets.K_branch, nets.K_trunk, nets.K_bias)
        return jax.nn.softplus(K_raw)  # softplus ensures K is always non-negative

    #generate the GBM model
    def generate_training_data(self, key, n_paths):
        dt = self.config.T / self.config.n_steps
        keys = jax.random.split(key, 5)

        S0 = jax.random.uniform(keys[0], (n_paths,), minval=self.config.S_lim[0],    maxval=self.config.S_lim[1])
        sigma = jax.random.uniform(keys[1], (n_paths,), minval=self.config.sigma_lim[0], maxval=self.config.sigma_lim[1])
        r = jax.random.uniform(keys[2], (n_paths,), minval=self.config.r_lim[0],     maxval=self.config.r_lim[1])

        # GBM (log-Euler scheme)
        dW = jax.random.normal(keys[3], (n_paths, self.config.n_steps)) * jnp.sqrt(dt)
        S_paths = jnp.zeros((n_paths, self.config.n_steps + 1))
        S_paths = S_paths.at[:, 0].set(S0)
        for i in range(self.config.n_steps):
            S_prev = S_paths[:, i]
            S_new = S_prev * jnp.exp((r - 0.5 * sigma**2) * dt + sigma * dW[:, i])
            S_paths = S_paths.at[:, i + 1].set(S_new)

        t_grid = jnp.linspace(0, self.config.T, self.config.n_steps + 1)
        return {'S_paths': S_paths, 't_grid': t_grid, 'sigma': sigma, 'r': r, 'dt': dt, 'dW': dW}

    def prepare_training_inputs(self, data):
    
        S_paths = data['S_paths']
        t_grid = data['t_grid']
        sigma = data['sigma']
        r = data['r']
        n_paths, n_steps_plus_1 = S_paths.shape

        S0 = S_paths[:, 0]
        S_hat = S_paths / S0[:, None]
        K_hat = self.config.K / S0

        # Trunk input: [t, S_hat, sigma, r]
        t_flat = jnp.repeat(t_grid[None, :], n_paths, axis=0).flatten()
        S_hat_flat = S_hat.flatten()
        sigma_flat = jnp.repeat(sigma[:, None], n_steps_plus_1, axis=1).flatten()
        r_flat = jnp.repeat(r[:, None], n_steps_plus_1, axis=1).flatten()
        trunk_input = jnp.stack([t_flat, S_hat_flat, sigma_flat, r_flat], axis=1)

        # Branch input: f(s_hat; K_hat) = max(s_hat - K_hat, 0) evaluated at sensor points
        sensor_points_hat = jnp.linspace(self.config.shat_min, self.config.shat_max, self.config.n_sensors)
        branch_per_path = jnp.maximum(sensor_points_hat[None, :] - K_hat[:, None], 0.0)
        branch_per_path = branch_per_path[:, None, :, None]
        branch_input = jnp.repeat(branch_per_path, n_steps_plus_1, axis=1)\
                         .reshape(-1, self.config.n_sensors, 1)
        return branch_input, trunk_input
# compute the loss function
    def compute_rbsde_loss(self, nets: DeepONetArchitecture, branch_input, trunk_input, data):
        
        S_paths = data['S_paths']
        r = data['r']
        sigma = data['sigma']
        dt = data['dt']
        dW = data['dW']

        n_paths, n_steps_plus_1 = S_paths.shape
        n_steps = self.config.n_steps
        S0 = S_paths[:, 0]
        S_hat = S_paths / S0[:, None]
        K_hat = self.config.K / S0
        L_hat = self.payoff_hat(S_hat, K_hat[:, None])

        # Get predictions from networks
        Y_all = self.compute_Yhat(branch_input, trunk_input, nets)
        Z_all = self.compute_Z(branch_input, trunk_input, nets)
        K_all = self.compute_Khat(branch_input, trunk_input, nets)

        Y = Y_all.reshape(n_paths, n_steps_plus_1)
        Z = Z_all.reshape(n_paths, n_steps_plus_1)
        K = K_all.reshape(n_paths, n_steps_plus_1)

        # Path weights based on moneyness (ITM, ATM, OTM)
        itm_mask = S0 > self.config.K
        atm_mask = jnp.abs(S0 - self.config.K) <= self.config.atm_range
        path_w = jnp.full(n_paths, self.config.otm_weight)
        path_w = jnp.where(itm_mask, self.config.itm_weight, path_w)
        path_w = jnp.where(atm_mask, self.config.atm_weight, path_w)
        path_w_col = path_w[:, None]

        # (1) Terminal loss: Y_T = L_T
        terminal_payoff_hat = L_hat[:, -1]
        terminal_loss = jnp.mean(path_w * (Y[:, -1] - terminal_payoff_hat) ** 2)

        # (2a) Local step residual: dY = ...
        Y_i, Y_ip1 = Y[:, :-1], Y[:, 1:]
        Z_i = Z[:, :-1]
        dK_ip1 = K[:, 1:] - K[:, :-1]
        Z_i_over_S0 = Z_i / S0[:, None]
        step_rhs = Y_i - r[:, None] * Y_i * dt + Z_i_over_S0 * dW + dK_ip1
        step_residual = Y_ip1 - step_rhs
        rbsde_step_loss = jnp.mean(path_w_col * (step_residual ** 2))

        # (2b) Global consistency from T
        r_reshaped = r[:, None]
        Y_T, K_T = Y[:, -1], K[:, -1]
        terms = []
        for i in range(n_steps):
            f_sum = jnp.sum((-r_reshaped * Y[:, i:n_steps]) * dt, axis=1)
            Z_sum = jnp.sum((Z[:, i:n_steps] / S0[:, None]) * dW[:, i:n_steps], axis=1)
            K_change = K_T - K[:, i]
            Y_from_T = Y_T - f_sum - Z_sum - K_change
            resid = path_w * (Y[:, i] - Y_from_T) ** 2
            terms.append(jnp.mean(resid))
        rbsde_global_loss = jnp.mean(jnp.array(terms))

        # (3) Obstacle constraint: Y >= L
        obstacle_violation = jnp.maximum(0.0, L_hat - Y)
        obstacle_loss = jnp.mean(path_w_col * obstacle_violation ** 2)

        # (4) Complementarity condition: (Y - L) dK = 0
        comp_accum = 0.0
        for i in range(n_steps):
            gap = jnp.maximum(Y[:, i] - L_hat[:, i], 0.0)
            dK_val = K[:, i + 1] - K[:, i]
            comp_accum += jnp.mean(path_w * (gap * dK_val) ** 2)
        complementarity_loss = comp_accum / n_steps

        # (5) Monotonicity of K: dK >= 0
        monotonicity_violation = jnp.maximum(0.0, K[:, :-1] - K[:, 1:])
        monotonicity_loss = jnp.mean(monotonicity_violation ** 2)

        # (6) K_0 approx 0, (7) K_T approx 0 
        initial_K_loss = jnp.mean(K[:, 0] ** 2)
        K_terminal_loss = jnp.mean(K[:, -1] ** 2)
        
        # New delta box loss
        denom = jnp.maximum(sigma[:, None] * jnp.maximum(S_paths, 1e-6), 1e-6)
        Delta = Z / denom
        delta_lower_violation = jnp.maximum(0.0, -Delta)
        delta_upper_violation = jnp.maximum(0.0, Delta - 1.0)
        delta_box_loss = jnp.mean(delta_lower_violation**2 + delta_upper_violation**2)

        total_loss = (
            self.config.terminal_weight * terminal_loss
            + self.config.rbsde_step_weight * rbsde_step_loss
            + self.config.rbsde_global_weight * rbsde_global_loss
            + self.config.obstacle_penalty * obstacle_loss
            + self.config.complementarity_penalty * complementarity_loss
            + self.config.monotonicity_weight * monotonicity_loss
            + self.config.initial_K_weight * initial_K_loss
            + self.config.k_terminal_weight * K_terminal_loss
            + self.config.z_box_weight * delta_box_loss
        )

        loss_dict = {
            'total': total_loss,
            'terminal': terminal_loss,
            'rbsde_step': rbsde_step_loss,
            'rbsde_global': rbsde_global_loss,
            'obstacle_metric': obstacle_loss,
            'complementarity': complementarity_loss,
            'monotonicity': monotonicity_loss,
            'initial_K': initial_K_loss,
            'K_terminal': K_terminal_loss,
            'delta_box': delta_box_loss,
        }
        return total_loss, loss_dict
#main training loop
    def train(self, key):
        key_init, key_train = jax.random.split(key)
        networks = DeepONetArchitecture.build_mlp_networks(self.config, key_init)
        optimizer = optax.adam(learning_rate=self.config.learning_rate)
        params, static = eqx.partition(networks, eqx.is_array)
        opt_state = optimizer.init(params)

        @eqx.filter_jit
        def make_step(params, opt_state, subkey, static):
            data = self.generate_training_data(subkey, self.config.batch_size)
            branch_input, trunk_input = self.prepare_training_inputs(data)

            def loss_fn(p):
                nets = eqx.combine(p, static)
                loss, _ = self.compute_rbsde_loss(nets, branch_input, trunk_input, data)
                return loss

            loss, grads = eqx.filter_value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            nets = eqx.combine(params, static)
            _, loss_dict = self.compute_rbsde_loss(nets, branch_input, trunk_input, data)
            return loss, params, opt_state, loss_dict

        losses = []
        loss_components = []

        print(" Starting RBSDE-DeepONet training (normalized by S0)...")
        print(f"Option Weights: ITM={self.config.itm_weight}, ATM={self.config.atm_weight} (range={self.config.atm_range}), OTM={self.config.otm_weight}")

        for iteration in range(self.config.n_iterations):
            key_train, subkey = jax.random.split(key_train)
            loss, params, opt_state, loss_dict = make_step(params, opt_state, subkey, static)
            losses.append(float(loss))
            loss_components.append({k: float(v) for k, v in loss_dict.items()})
            if iteration % 200 == 0 or iteration == self.config.n_iterations - 1:
                print(f"Iteration {iteration}: Total Loss = {float(loss):.6f}")
                print(f"  Components: terminal={loss_dict['terminal']:.4e}, "
                      f"step={loss_dict['rbsde_step']:.4e}, global={loss_dict['rbsde_global']:.4e}, "
                      f"obstacle={loss_dict['obstacle_metric']:.4e}, "
                      f"delta_box={loss_dict['delta_box']:.4e}")

        final_networks = eqx.combine(params, static)
        return final_networks, {'losses': losses, 'loss_components': loss_components, 'final_loss': losses[-1] if losses else float('inf')}

    def evaluate(self, networks: DeepONetArchitecture, test_key, n_test=400):
        data = self.generate_training_data(test_key, n_test)
        S0 = data['S_paths'][:, 0]
        r_values = data['r']
        sigma_values = data['sigma']
        K_hat = self.config.K / S0

        # t=0 inputs
        t0 = jnp.zeros_like(S0)
        s_hat0 = jnp.ones_like(S0)
        trunk_input_t0 = jnp.stack([t0, s_hat0, sigma_values, r_values], axis=1)

        sensor_points_hat = jnp.linspace(self.config.shat_min, self.config.shat_max, self.config.n_sensors)
        branch_per_path = jnp.maximum(sensor_points_hat[None, :] - K_hat[:, None], 0.0)
        branch_input_t0 = branch_per_path[:, :, None]

        # Predict prices and deltas from DeepONet
        Y_hat_raw = self.compute_Yhat(branch_input_t0, trunk_input_t0, networks)
        Y_hat_floor = self.payoff_hat(1.0, K_hat)
        Y_hat = jnp.maximum(Y_hat_raw, Y_hat_floor)
        deeponet_prices = Y_hat * S0
        price_norm_dn = Y_hat

        Z_at_t0 = self.compute_Z(branch_input_t0, trunk_input_t0, networks)
        denom = jnp.maximum(sigma_values * jnp.maximum(S0, 1e-6), 1e-6)
        delta_unclipped = Z_at_t0 / denom
        deeponet_deltas = jnp.clip(delta_unclipped, 0.0, 1.0)
        Z_from_delta_dn = deeponet_deltas * sigma_values * S0

        # Black-Scholes baseline
        bs_prices = jax.vmap(lambda s, r, sig: self.euro.price(s, self.config.K, self.config.T, r, sig))(S0, r_values, sigma_values)
        bs_deltas = jax.vmap(lambda s, r, sig: self.euro.delta(s, self.config.K, self.config.T, r, sig))(S0, r_values, sigma_values)
        price_norm_bs = bs_prices / jnp.maximum(S0, 1e-6)
        Z_bs = bs_deltas * sigma_values * S0

        american_premium = deeponet_prices - bs_prices
        abs_err = jnp.abs(deeponet_prices - bs_prices)
        rel_err_s0 = rel_err_vs_s0(deeponet_prices, bs_prices, S0) * 100.0
        safe_bs = jnp.maximum(bs_prices, 1e-6)
        rel_err_bs = (abs_err / safe_bs) * 100.0
        price_threshold = 0.1
        masked_rel_bs_mean = masked_rel_err(deeponet_prices, bs_prices, threshold=price_threshold) * 100.0
        smape_vs_bs = smape(deeponet_prices, bs_prices) * 100.0
        abs_err_norm = jnp.abs(price_norm_dn - price_norm_bs)
        rmse_norm_price = float(jnp.sqrt(jnp.mean((price_norm_dn - price_norm_bs) ** 2)))
        mae_norm_price = float(jnp.mean(abs_err_norm))
        price_corr_norm = float(jnp.corrcoef(price_norm_dn, price_norm_bs)[0, 1])
        rmse_delta = float(jnp.sqrt(jnp.mean((deeponet_deltas - bs_deltas) ** 2)))
        delta_corr_clipped = float(jnp.corrcoef(deeponet_deltas, bs_deltas)[0, 1])
        delta_corr_unclipped = float(jnp.corrcoef(delta_unclipped, bs_deltas)[0, 1])
        rmse_Z = float(jnp.sqrt(jnp.mean((Z_from_delta_dn - Z_bs) ** 2)))
        Z_corr = float(jnp.corrcoef(Z_from_delta_dn, Z_bs)[0, 1])
        itm_mask = S0 > self.config.K
        atm_mask = jnp.abs(S0 - self.config.K) <= self.config.atm_range
        otm_mask = S0 < self.config.K
        def bucket_stats(mask):
            m = jnp.where(mask)[0]
            if m.size == 0:
                return dict(count=0, mae=float('nan'), rmse=float('nan'), delta_rmse=float('nan'), premium_mean=float('nan'),
                            mae_norm=float('nan'), rmse_norm=float('nan'))
            mae = float(jnp.mean(jnp.abs(deeponet_prices[m] - bs_prices[m])))
            rmse = float(jnp.sqrt(jnp.mean((deeponet_prices[m] - bs_prices[m])**2)))
            d_rmse = float(jnp.sqrt(jnp.mean((deeponet_deltas[m] - bs_deltas[m])**2)))
            pm = float(jnp.mean(american_premium[m]))
            mae_n = float(jnp.mean(jnp.abs(price_norm_dn[m] - price_norm_bs[m])))
            rmse_n = float(jnp.sqrt(jnp.mean((price_norm_dn[m] - price_norm_bs[m]) ** 2)))
            return dict(count=int(m.size), mae=mae, rmse=rmse, delta_rmse=d_rmse, premium_mean=pm,
                        mae_norm=mae_n, rmse_norm=rmse_n)

        per_bucket = {
            "ITM": bucket_stats(itm_mask),
            "ATM": bucket_stats(atm_mask),
            "OTM": bucket_stats(otm_mask),
        }
        price_corr = float(jnp.corrcoef(deeponet_prices, bs_prices)[0, 1])
        violation_rate = float(jnp.mean(Y_hat_raw < Y_hat_floor))

        return {
            'deeponet_prices': deeponet_prices,
            'deeponet_prices_normalized': price_norm_dn,
            'deeponet_deltas': deeponet_deltas,
            'deeponet_Z0_from_delta': Z_from_delta_dn,
            'bs_prices': bs_prices,
            'bs_prices_normalized': price_norm_bs,
            'bs_deltas': bs_deltas,
            'bs_Z0': Z_bs,
            'american_premium': american_premium,
            'price_errors_vs_bs': abs_err,
            'relative_price_errors_vs_bs': rel_err_bs,
            'relative_price_errors_vs_S0': rel_err_s0,
            'statistics': {
                'non_normalized': {
                    'price': {'mean_abs_err': float(jnp.mean(abs_err)), 'rmse': float(jnp.sqrt(jnp.mean(abs_err ** 2))), 'correlation': price_corr, 'mean_rel_err_vs_bs_%': float(jnp.mean(rel_err_bs)), 'mean_rel_err_vs_bs_%_filtered': float(masked_rel_bs_mean), 'smape_vs_bs_%': float(smape_vs_bs)},
                    'delta': {'rmse': rmse_delta, 'corr_clipped': delta_corr_clipped, 'corr_unclipped': delta_corr_unclipped},
                    'Z': {'rmse': rmse_Z, 'correlation': Z_corr},
                    'american_premium_mean': float(jnp.mean(american_premium)),
                },
                'normalized': {
                    'price': {'mae': mae_norm_price, 'rmse': rmse_norm_price, 'correlation': price_corr_norm, 'mean_rel_err_vs_S0_%': float(jnp.mean(rel_err_s0))}
                },
                'mean_abs_err_vs_bs': float(jnp.mean(abs_err)), 'rmse_price_vs_bs': float(jnp.sqrt(jnp.mean(abs_err ** 2))),
                'mean_rel_err_vs_S0_%': float(jnp.mean(rel_err_s0)),
                'mean_rel_err_vs_bs_%': float(jnp.mean(rel_err_bs)),
                'mean_rel_err_vs_bs_%_filtered': float(masked_rel_bs_mean),
                'smape_vs_bs_%': float(smape_vs_bs),
                'mean_american_premium': float(jnp.mean(american_premium)),
                'violation_rate': violation_rate,
                'price_correlation': price_corr,
                'delta_correlation_clipped': delta_corr_clipped,
                'delta_correlation_unclipped': delta_corr_unclipped,
                'rmse_delta': rmse_delta,
                'per_bucket': per_bucket,
            },
            'test_data': {'S0_values': S0, 'r_values': r_values, 'sigma_values': sigma_values}
        }
    
    def premium_grid_r_fixed(self, networks: DeepONetArchitecture, S0_grid, sigma_grid, r_fixed):
        S0g, sigmag = jnp.meshgrid(S0_grid, sigma_grid, indexing='xy')
        S0_flat, sigma_flat = S0g.reshape(-1), sigmag.reshape(-1)
        r_flat = jnp.full_like(S0_flat, r_fixed)

        K_hat = self.config.K / S0_flat
        t0, s_hat0 = jnp.zeros_like(S0_flat), jnp.ones_like(S0_flat)
        trunk_input = jnp.stack([t0, s_hat0, sigma_flat, r_flat], axis=1)

        sensor_points_hat = jnp.linspace(self.config.shat_min, self.config.shat_max, self.config.n_sensors)
        branch_per_path = jnp.maximum(sensor_points_hat[None, :] - K_hat[:, None], 0.0)
        branch_input = branch_per_path[:, :, None]

        Y_hat_raw = self.compute_Yhat(branch_input, trunk_input, networks)
        Y_floor = self.payoff_hat(1.0, K_hat)
        Y0 = jnp.maximum(Y_hat_raw, Y_floor)
        DN_price = Y0 * S0_flat
        BS_price = jax.vmap(lambda s, r, sig: self.euro.price(s, self.config.K, self.config.T, r, sig))(S0_flat, r_flat, sigma_flat)
        
        premium = (DN_price - BS_price).reshape(S0g.shape)
        return S0g, sigmag, premium
    
    def premium_grid_sigma_fixed(self, networks: DeepONetArchitecture, S0_grid, r_grid, sigma_fixed):
        S0g, rg = jnp.meshgrid(S0_grid, r_grid, indexing='xy')
        S0_flat, r_flat = S0g.reshape(-1), rg.reshape(-1)
        sigma_flat = jnp.full_like(S0_flat, sigma_fixed)

        K_hat = self.config.K / S0_flat
        t0, s_hat0 = jnp.zeros_like(S0_flat), jnp.ones_like(S0_flat)
        trunk_input = jnp.stack([t0, s_hat0, sigma_flat, r_flat], axis=1)
        
        sensor_points_hat = jnp.linspace(self.config.shat_min, self.config.shat_max, self.config.n_sensors)
        branch_per_path = jnp.maximum(sensor_points_hat[None, :] - K_hat[:, None], 0.0)
        branch_input = branch_per_path[:, :, None]

        Y_hat_raw = self.compute_Yhat(branch_input, trunk_input, networks)
        Y_floor = self.payoff_hat(1.0, K_hat)
        Y0 = jnp.maximum(Y_hat_raw, Y_floor)
        DN_price = Y0 * S0_flat
        BS_price = jax.vmap(lambda s, r, sig: self.euro.price(s, self.config.K, self.config.T, r, sig))(S0_flat, r_flat, sigma_flat)
        
        premium = (DN_price - BS_price).reshape(S0g.shape)
        return S0g, rg, premium
