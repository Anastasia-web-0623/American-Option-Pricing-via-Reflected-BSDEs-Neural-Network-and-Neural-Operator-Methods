import math
from typing import Tuple,Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import norm

#This defined the whole structure of two neural networks, y0 and z in each time step

class RBSDENet(nn.Module):
    def __init__(self,d_input = 2,d_hidden = 128,n_layers = 4):
        super().__init__()
        #y0 network:using sigmoid to ensure range (0,1)
        self.y0_net = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden//2),
            nn.ReLU(),
            nn.Linear(d_hidden//2, 1),
            nn.Sigmoid()
        )
        # z network
        act = nn.SiLU() # using SiLU to ensure better perfomance than using ReLU
        z_layers = [nn.Linear(d_input, d_hidden), nn.LayerNorm(d_hidden), act]
        for _ in range(n_layers - 2): #since z will conduct in each time setp
            z_layers += [nn.Linear(d_hidden, d_hidden), nn.LayerNorm(d_hidden), act]
        z_layers += [nn.Linear(d_hidden, 1)]
        self.z_net = nn.Sequential(*z_layers)
        self.softplus = nn.Softplus(beta=1.5)
        self._init()
        
    #Xavier initialization for network weight
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
     #compute the normalized y0,here k_rel = k/s0
    def y0_norm(self, k_rel: float, device: str) -> torch.Tensor:
        t0 = torch.zeros(1, 1, device=device)
        log_x_over_k = torch.tensor([[math.log(1.0 / k_rel)]], device=device)
        inp = torch.cat([t0, log_x_over_k], dim=-1)
        return self.y0_net(inp)
        
    # Computes the normalized Z value
    def z_norm(self, t_norm: torch.Tensor, x: torch.Tensor, k_rel: float) -> torch.Tensor:
        log_x_over_k = torch.log(x / k_rel)
        z_inp = torch.cat([t_norm, log_x_over_k], dim=-1)
        z_raw = self.z_net(z_inp)
        return self.softplus(z_raw) + 1e-8 #the final value to ensure this should be positive

# This class will contain simulation paths and run the forward, loss to finish training

class AmericanCallRBSDE:
    def __init__(self, S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, q=0.0, n_steps=160, device='cpu', use_antithetic=True):
        self.S0 = float(S0); self.K = float(K); self.r = float(r)
        self.sigma = float(sigma); self.T = float(T); self.q = float(q)
        self.n_steps = int(n_steps); self.dt = self.T / self.n_steps
        self.device = device; self.use_antithetic = bool(use_antithetic)
        self.k_rel = self.K / self.S0
        self.net = RBSDENet().to(device)
        
        #Loss weight, this is calculated by using random search to find the best combinations which is specific to this problem
        self.w_y0 = 60.087281985351794
        self.w_mart = 1.0
        self.w_comp = 7.951968987745549
        self.lambda_k = 0.0009204665464386782
        self.w_y0_refine = 20.0
        self.w_mart_refine = 1.6069195692361902
        self.w_comp_refine = 19.879922469363873
        self.x_paths_cache = None
        self.dW_cache = None
    #payoff paths
    def payoff_norm(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x - self.k_rel, min=0.0)
    
    #this is the driver function of RBSDE
    def generator_f_norm(self, y: torch.Tensor) -> torch.Tensor:
        return self.r * y

    #this is the simulated asset price paths
    def simulate_paths_norm(self, batch_size: int):
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
    
    #forward pass of RBSDE solver
    def rbsde_forward(self,batch_size:int):
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
        payoff_T_norm = self.payoff_norm(x[:,-1:])
        return payoff_T_norm, y0_norm,y_path, push_path,tgrid
    
    # the loss function
    def compute_loss(self,batch_size:int,weights:dict):
        payoff_T_norm, y0_norm, y_path, push_path, _ = self.rbsde_forward(batch_size)
        y_T = y_path[-1]  # terminal
        terminal_loss = torch.mean((payoff_T_norm - y_T) ** 2)
        # y0 target
        disc_T = math.exp(-self.r * self.T)
        disc_push = torch.tensor(0.0, device=self.device)
        for i in range(self.n_steps):
            t_next = (i + 1) * self.dt
            disc_push = disc_push + math.exp(-self.r * t_next) * push_path[i].mean()
        y0_target_norm = disc_T * payoff_T_norm.mean() + disc_push
        y0_loss = (y0_norm.mean() - y0_target_norm) ** 2

        # reflection loss 
        avg_push = torch.mean(torch.stack([p.mean() for p in push_path]))
        reflection_loss = weights['lambda_k'] * avg_push

        # martingale increment loss 
        inc_penalties = []
        for i in range(self.n_steps):
            t_i = i * self.dt; t_ip1 = (i + 1) * self.dt
            disc_i = math.exp(-self.r * t_i); disc_ip1 = math.exp(-self.r * t_ip1)
            y_i = y_path[i]; y_ip1 = y_path[i + 1]; push_i = push_path[i]
            dM_i = disc_ip1 * y_ip1 - disc_i * y_i + disc_ip1 * push_i
            inc_penalties.append((dM_i ** 2).mean())
        martingale_inc_loss = torch.stack(inc_penalties).mean()

        # complementarity loss
        comp_penalties = []
        for i in range(self.n_steps):
            L_next = self.payoff_norm(self.x_paths_cache[:, i+1:i+2])
            y_next = y_path[i + 1]
            push_i = push_path[i]
            slack = torch.relu(y_next - L_next)
            comp_penalties.append((push_i * slack).mean())
        complementarity_loss = torch.stack(comp_penalties).mean()

        total_loss = (terminal_loss + reflection_loss + 
                      weights['w_y0'] * y0_loss + 
                      weights['w_mart'] * martingale_inc_loss + 
                      weights['w_comp'] * complementarity_loss)
        
        return (total_loss, terminal_loss, reflection_loss, y0_loss, 
                martingale_inc_loss, complementarity_loss, y0_target_norm)
    
    # main train loop
    def train(self, n_epochs=5000, batch_size=384, lr=8e-4, verbose_every=250, patience=5000):
        optimizer = optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)

        best_loss = float('inf'); patience_counter = 0

        # Use refinement weights for training
        weights = dict(w_y0=self.w_y0_refine, w_mart=self.w_mart_refine, w_comp=self.w_comp_refine, lambda_k=self.lambda_k)
        
        print(f"start training (normalized); T={self.T:.2f}, n_steps={self.n_steps}, q={self.q:.4f}")
        for epoch in range(n_epochs):
            self.net.train(); optimizer.zero_grad()
            (total_loss, terminal_loss, reflection_loss, y0_loss, mart_loss, comp_loss, y0_target_norm) = self.compute_loss(batch_size, weights=weights)

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
                      f"Reflect={reflection_loss.item():.6f}, Mart={mart_loss.item():.6f}, Comp={comp_loss.item():.6f}, "
                      f"Price={price_abs:.4f} (norm={price_norm:.6f}), y0_pred={price_norm:.6f}, "
                      f"y0_target={y0_target_norm.item():.6f}, LR={curr_lr:.6f}")
            if patience_counter > patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Get the predicted option price
    def get_option_price(self) -> Tuple[float, float]:
        self.net.eval()
        with torch.no_grad():
            y0_norm = self.net.y0_norm(self.k_rel, self.device).item()
            return self.S0 * y0_norm, y0_norm

    # Predict t=0 price and delta
    @torch.no_grad()
    def predict_t0(self) -> Dict[str, float]:
        # Calculate price from y0
        y0 = self.net.y0_norm(self.k_rel, self.device).item()
        price = self.S0 * y0
        
        # Calculate delta from Z0
        t_norm = torch.zeros(1, 1, device=self.device)
        x1 = torch.ones(1, 1, device=self.device)
        Z0 = self.net.z_norm(t_norm, x1, self.k_rel).item()
        denom = max(self.sigma, 1e-8)
        delta_unclipped = Z0 / denom
        delta = float(min(max(delta_unclipped, 0.0), 1.0))
        
        # Black-Scholes baseline
        d1 = (np.log(self.S0 / self.K) + (self.r - self.q + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        bs_price = self.S0 * np.exp(-self.q * self.T) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        bs_price = max(bs_price, 0.0)

        if self.T <= 1e-12:
            bs_delta = float(1.0 if self.S0 >= self.K else 0.0)
        else:
            d1_bs = (np.log(self.S0 / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
            Nd1 = 0.5 * (1.0 + math.erf(d1_bs / math.sqrt(2.0)))
            bs_delta = float(np.exp(-self.q * self.T) * Nd1)

        return dict(price=price, price_norm=y0, delta=delta, bs_price=bs_price, bs_delta=bs_delta, premium=price-bs_price)



