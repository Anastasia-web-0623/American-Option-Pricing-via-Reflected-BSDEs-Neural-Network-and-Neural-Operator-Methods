import torch
import torch.nn as nn
import math

class RBSDENet(nn.Module):
    def __init__(self, d_input=2, d_hidden=128, n_layers=4):
        super().__init__()
        self.y0_net = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden//2),
            nn.ReLU(),
            nn.Linear(d_hidden//2, 1),
            nn.Sigmoid()
        )

        act = nn.SiLU()
        z_layers = [nn.Linear(d_input, d_hidden), nn.LayerNorm(d_hidden), act]
        for _ in range(n_layers - 2):
            z_layers += [nn.Linear(d_hidden, d_hidden), nn.LayerNorm(d_hidden), act]
        z_layers += [nn.Linear(d_hidden, 1)]
        self.z_net = nn.Sequential(*z_layers)
        self.softplus = nn.Softplus(beta=1.5)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def y0_norm(self, k_rel: float, device: str) -> torch.Tensor:
        t0 = torch.zeros(1, 1, device=device)
        log_x_over_k = torch.tensor([[math.log(1.0 / k_rel)]], device=device)
        inp = torch.cat([t0, log_x_over_k], dim=-1)
        return self.y0_net(inp)

    def z_norm(self, t_norm: torch.Tensor, x: torch.Tensor, k_rel: float) -> torch.Tensor:
        log_x_over_k = torch.log(x / k_rel)
        z_inp = torch.cat([t_norm, log_x_over_k], dim=-1)
        z_raw = self.z_net(z_inp)
        return self.softplus(z_raw) + 1e-8