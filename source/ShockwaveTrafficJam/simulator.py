import os

import torch

max_acceleration = 10.0  # maximum acceleration for the vehicles (m/s^2)
max_deceleration = 100.0  # maximum deceleration for the vehicles (m/s^2)
collision_threshold = 0.1  # threshold for collision detection


class databatch:
    def __init__(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        len_road: torch.Tensor,
        d0_OVM: torch.Tensor,
        Delta: torch.Tensor,
        Vmax: torch.Tensor,
        tau: torch.Tensor,
        d0_FTL: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ):
        """Traffic data batch.

        Args:
            x: position of the vehicles on the road (0 <= x < len_road) for each batch element
                it is a tensor of size (batch_size, n_vehicle)
            v: velocity of the vehicles (0 <= v <= Vmax) for each batch element
                it is a tensor of size (batch_size, n_vehicle)
            len_road: length of the road (0 <= len_road)
                it is a tensor of size (batch_size)
            d0_OVM: minimum distance between vehicles for OVM (0 <= d0_OVM)
                it is a tensor of size (batch_size)
            Delta: parameter for OVM (0 < Delta)
                it is a tensor of size (batch_size)
            Vmax: maximum velocity of the vehicles (0 <= Vmax)
                it is a tensor of size (batch_size)
            tau: parameter for OVM (0 < tau)
                it is a tensor of size (batch_size)
            d0_FTL: minimum distance between vehicles for FTL (0 < d0_FTL)
                it is a tensor of size (batch_size)
            gamma: parameter for FTL (0 <= gamma)
                it is a tensor of size (batch_size)
            beta: weight for OVM and FTL (0 <= alpha)
                it is a tensor of size (batch_size)
        """
        self.x = x
        self.v = v
        self.len_road = len_road.unsqueeze(1)
        self.d0_OVM = d0_OVM.unsqueeze(1)
        self.Delta = Delta.unsqueeze(1)
        self.Vmax = Vmax.unsqueeze(1)
        self.tau = tau.unsqueeze(1)
        self.d0_FTL = d0_FTL.unsqueeze(1)
        self.gamma = gamma.unsqueeze(1)
        self.beta = beta.unsqueeze(1)

        self.dtype = x.dtype
        self.device = x.device
        self.shape = (x.size(0), x.size(1))

    def FTL(self, d_x: torch.Tensor, d_v: torch.Tensor) -> torch.Tensor:
        return d_v / ((d_x / self.d0_FTL) ** (1 + self.gamma))

    def OVM(self, d_x: torch.Tensor) -> torch.Tensor:
        V = (
            self.Vmax
            * (
                torch.tanh((d_x - self.d0_OVM) / self.Delta)
                + torch.tanh(self.d0_OVM / self.Delta)
            )
            / (1 + torch.tanh(self.d0_OVM / self.Delta))
        )
        return (self.tau) ** (-1) * (V - self.v)

    def F(self, x, v):
        d_x = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        d_v = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        d_x[:, :-1] = x[:, 1:] - x[:, :-1]
        d_x[:, -1] = x[:, 0] - x[:, -1]
        d_x = torch.remainder(d_x, self.len_road)
        d_v[:, :-1] = v[:, 1:] - v[:, :-1]
        d_v[:, -1] = v[:, 0] - v[:, -1]
        return (self.beta * self.OVM(d_x) + self.FTL(d_x, d_v)) / (1 + self.beta)

    def auto_step(self, dt: float):
        # apply heun method

        # calcolo la forza
        F = self.F(self.x, self.v)
        torch.clamp(F, -max_deceleration, max_acceleration, out=F)

        # f = [v,F]

        d_x = self.v * dt
        d_v = F * dt

        # calcolo nuovamente la forza
        F_heun = self.F(self.x + d_x, self.v + d_v)
        torch.clamp(F, -max_deceleration, max_acceleration, out=F_heun)

        # f_tilde = [self.v + d_v, F_heun]

        d_x = (dt / 2) * (2 * self.v + d_v)
        d_v = (dt / 2) * (F + F_heun)

        # aggiorno
        self.x += d_x
        self.v += d_v

        # impongo i vincoli
        torch.remainder(self.x, self.len_road, out=self.x)
        torch.clamp(self.v, torch.zeros_like(self.Vmax), self.Vmax, out=self.v)

        # nan check
        if torch.isnan(self.x).any() or torch.isnan(self.v).any():
            raise ValueError("Nan detected")

    def auto_simulate(self, dt: float, n_steps: int):
        for _ in range(n_steps):
            self.auto_step(dt)

    def visual(self):
        import plotly.graph_objects as go

        # calcolo il raggio della strada
        r = self.len_road / (2 * 3.1415)

        theta = (self.x / self.len_road) * (2 * 3.1415)
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        # mostro gli scatter in molteplici grafici distinti senza mettere la legenda
        fig = go.Figure()
        for i in range(self.shape[0]):
            fig.add_trace(
                go.Scatter(x=x[i].cpu().numpy(), y=y[i].cpu().numpy(), mode="markers")
            )
        fig.update_layout(width=800, height=800)
        fig.show()

    def to(self, device: torch.device):
        self.x = self.x.to(device)
        self.v = self.v.to(device)
        self.len_road = self.len_road.to(device)
        self.d0_OVM = self.d0_OVM.to(device)
        self.Delta = self.Delta.to(device)
        self.Vmax = self.Vmax.to(device)
        self.tau = self.tau.to(device)
        self.d0_FTL = self.d0_FTL.to(device)
        self.gamma = self.gamma.to(device)
        self.beta = self.beta.to(device)
        self.device = device
        return self

    def save(self, index: int, path: str):
        # concateno tutti i tensori in un solo tensore
        self_tensor = torch.cat(
            [
                self.x,
                self.v,
                self.len_road,
                self.d0_OVM,
                self.Delta,
                self.Vmax,
                self.tau,
                self.d0_FTL,
                self.gamma,
                self.beta,
            ],
            dim=1,
        )
        torch.save(
            self_tensor,
            os.path.join(path, f"ShockwaveTrafficJam_{index}_{self.shape}.pt"),
        )
