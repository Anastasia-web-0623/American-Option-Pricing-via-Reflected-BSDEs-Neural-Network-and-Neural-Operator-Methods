import math
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm
from typing import Tuple, Dict
from rbsde_net import RBSDENet

class AmericanCallRBSDE:
    def __init__(self, S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, q=0.0, n_steps=160, device='cpu', use_antithetic=True):
        self.S0 = float(S0); self.K = float(K); self.r = float(r)
        self.sigma = float(sigma); self.T = float(T); self.q = float(q)
        self.n_steps = int(n_steps); self.dt = self.T / self.n_steps
        self.device = device; self.use_antithetic = bool(use_antithetic)
        self.k_rel = self.K / self.S0
        self.net = RBSDENet().to(device)

        # loss weights
        self.w_y0 = 60.087281985351794
        self.w_mart = 1.0
        self.w_comp = 7.951968987745549
        self.w_delta = 10.0
        self.lambda_k = 0.0009204665464386782

        self.w_y0_refine = 20.0
        self.w_mart_refine = 1.6069195692361902
        self.w_comp_refine = 19.879922469363873
        self.w_delta_refine = 100.0
        
        self.x_paths_cache = None
        self.dW_cache = None

    def black_scholes_call_price(self, S, K, T, r, sigma, q=0.0):
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        call_price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        return max(call_price, 0.0)

    def black_scholes_delta(self, S, K, T, r, sigma, q=0.0) -> float:
        if T <= 1e-12:
            return float(1.0 if S >= K else 0.0)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        Nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
        return float(math.exp(-q * T) * Nd1)

    def payoff_norm(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x - self.k_rel, min=0.0)

    def generator_f_norm(self, y: torch.Tensor) -> torch.Tensor:
        return self.r * y

    def simulate_paths_norm(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_grid = torch.linspace(0, self.T, self.n_steps + 1, device=self.device)
        if self.use_antithetic:
            half = batch_size // 2
            dW_half = torch.randn(half, self.n_steps, device=self.device) * math.sqrt(self.dt)
            dW = torch.cat([dW_half, -dW_half], dim=0)
            if batch_size % 2 == 1:
                extra = torch.randn(1, self.n_steps, device=self.device) * math.sqrt(self.dt)
                dW = torch.cat([dW, extra], dim=0)
        else:
            dW = torch.randn(batch_size, self.n_steps, device=self.device) * math.sqrt(self.dt)
            
        x = torch.zeros(dW.size(0), self.n_steps + 1, device=self.device)
        x[:, 0] = 1.0
        drift = (self.r - self.q) - 0.5 * self.sigma ** 2
        for i in range(self.n_steps):
            x[:, i + 1] = x[:, i] * torch.exp(drift * self.dt + self.sigma * dW[:, i])
        return time_grid, dW, x

    def rbsde_forward(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, list, list, torch.Tensor]:
        tgrid, dW, x = self.simulate_paths_norm(batch_size)
        self.x_paths_cache = x; self.dW_cache = dW
        y_path = []; push_path = []
            
        y0_norm = self.net.y0_norm(self.k_rel, self.device)
        y = y0_norm.repeat(x.size(0), 1)
        y_path.append(y)

        for i in range(self.n_steps):
            ti = tgrid[i].item()
            t_norm = torch.full((x.size(0), 1), ti / self.T, device=self.device)
            xi = x[:, i:i+1]
            z_norm = self.net.z_norm(t_norm, xi, self.k_rel)
            f_val = self.generator_f_norm(y)
            y_tilde = y + (-f_val * self.dt + z_norm * dW[:, i:i+1])
            L_next = self.payoff_norm(x[:, i+1:i+2])
                
            push = torch.relu(L_next - y_tilde)
            y = y_tilde + push
                
            y_path.append(y.clone())
            push_path.append(push)

        payoff_T_norm = self.payoff_norm(x[:, -1:])
        return payoff_T_norm, y0_norm, y_path, push_path, tgrid

    def compute_delta_loss(self, batch_size: int) -> torch.Tensor:
        tgrid, _, x = self.simulate_paths_norm(batch_size)
        t0 = tgrid[0].item()
        x0 = x[:, 0:1]
        t_norm = torch.full((batch_size, 1), t0 / self.T, device=self.device)
        z_norm = self.net.z_norm(t_norm, x0, self.k_rel)
        pred_delta_norm = z_norm / self.sigma
        bs_delta_target = self.black_scholes_delta(self.S0, self.K, self.T, self.r, self.sigma, self.q)
        bs_delta_target_tensor = torch.full_like(pred_delta_norm, bs_delta_target, device=self.device)
        delta_loss = torch.mean((pred_delta_norm - bs_delta_target_tensor) ** 2)
        return delta_loss

    def compute_loss(self, batch_size: int, weights: dict) -> Tuple[torch.Tensor, ...]:
        payoff_T_norm, y0_norm, y_path, push_path, _ = self.rbsde_forward(batch_size)
        y_T = y_path[-1]

        terminal_loss = torch.mean((payoff_T_norm - y_T) ** 2)
        disc_T = math.exp(-self.r * self.T)
        disc_push = torch.tensor(0.0, device=self.device)
        for i in range(self.n_steps):
            t_next = (i + 1) * self.dt
            disc_push = disc_push + math.exp(-self.r * t_next) * push_path[i].mean()
        y0_target_norm = disc_T * payoff_T_norm.mean() + disc_push
        y0_loss = (y0_norm.mean() - y0_target_norm) ** 2

        avg_push = torch.mean(torch.stack([p.mean() for p in push_path]))
        reflection_loss = weights['lambda_k'] * avg_push

        inc_penalties = []
        for i in range(self.n_steps):
            t_i = i * self.dt; t_ip1 = (i + 1) * self.dt
            disc_i = math.exp(-self.r * t_i); disc_ip1 = math.exp(-self.r * t_ip1)
            y_i = y_path[i]; y_ip1 = y_path[i + 1]; push_i = push_path[i]
            dM_i = disc_ip1 * y_ip1 - disc_i * y_i + disc_ip1 * push_i
            inc_penalties.append((dM_i ** 2).mean())
        martingale_inc_loss = torch.stack(inc_penalties).mean()

        comp_penalties = []
        for i in range(self.n_steps):
            L_next = self.payoff_norm(self.x_paths_cache[:, i+1:i+2])
            y_next = y_path[i + 1]
            push_i = push_path[i]
            slack = torch.relu(y_next - L_next)
            comp_penalties.append((push_i * slack).mean())
        complementarity_loss = torch.stack(comp_penalties).mean()

        delta_loss = self.compute_delta_loss(batch_size)

        total_loss = (terminal_loss + reflection_loss + 
                      weights['w_y0'] * y0_loss + 
                      weights['w_mart'] * martingale_inc_loss + 
                      weights['w_comp'] * complementarity_loss +
                      weights['w_delta'] * delta_loss)
            
        return (total_loss, terminal_loss, reflection_loss, y0_loss, 
                martingale_inc_loss, complementarity_loss, delta_loss, y0_target_norm)

    def train(self, n_epochs=5000, batch_size=384, lr=8e-4, verbose_every=250, patience=5000):
        optimizer = optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)

        best_loss = float('inf'); patience_counter = 0

        weights = dict(w_y0=self.w_y0_refine, w_mart=self.w_mart_refine, w_comp=self.w_comp_refine, 
                        w_delta=self.w_delta_refine, lambda_k=self.lambda_k)
            
        print(f"start training (normalized); T={self.T:.2f}, n_steps={self.n_steps}, q={self.q:.4f}")
        for epoch in range(n_epochs):
            self.net.train(); optimizer.zero_grad()
                
            (total_loss, terminal_loss, reflection_loss, y0_loss, mart_loss, comp_loss, delta_loss, y0_target_norm) = self.compute_loss(batch_size, weights=weights)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step(total_loss)

            if total_loss.item() < best_loss:
                best_loss = total_loss.item(); patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % verbose_every == 0:
                price_abs, price_norm = self.get_option_price()
                curr_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{n_epochs}: Total={total_loss.item():.6f}, "
                      f"Terminal={terminal_loss.item():.6f}, y0={y0_loss.item():.6f}, "
                      f"Reflect={reflection_loss.item():.6f}, Mart={mart_loss.item():.6f}, Comp={comp_loss.item():.6f}, Delta={delta_loss.item():.6f}, "
                      f"Price={price_abs:.4f} (norm={price_norm:.6f}), y0_pred={price_norm:.6f}, "
                      f"y0_target={y0_target_norm.item():.6f}, LR={curr_lr:.6f}")

            if patience_counter > patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    def get_option_price(self) -> Tuple[float, float]:
        self.net.eval()
        with torch.no_grad():
            y0_norm = self.net.y0_norm(self.k_rel, self.device).item()
            return self.S0 * y0_norm, y0_norm

    @torch.no_grad()
    def predict_t0(self) -> Dict[str, float]:
        y0 = self.net.y0_norm(self.k_rel, self.device).item()
        price = self.S0 * y0
            
        t_norm = torch.zeros(1, 1, device=self.device)
        x1 = torch.ones(1, 1, device=self.device)
        Z0 = self.net.z_norm(t_norm, x1, self.k_rel).item()
        denom = max(self.sigma, 1e-8)
        delta_unclipped = Z0 / denom
        delta = float(min(max(delta_unclipped, 0.0), 1.0))

        bs = self.black_scholes_call_price(self.S0, self.K, self.T, self.r, self.sigma, self.q)
        bs_delta = self.black_scholes_delta(self.S0, self.K, self.T, self.r, self.sigma, self.q)

        return dict(price=price, price_norm=y0, delta=delta, bs_price=bs, bs_delta=bs_delta, premium=price-bs)