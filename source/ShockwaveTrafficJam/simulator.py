import os

import torch

max_acceleration = 18.0  # maximum acceleration for the vehicles (m/s^2)
max_deceleration = 180.0  # maximum deceleration for the vehicles (m/s^2)


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
        self.len_road = len_road.reshape(-1, 1)
        self.d0_OVM = d0_OVM.reshape(-1, 1)
        self.Delta = Delta.reshape(-1, 1)
        self.Vmax = Vmax.reshape(-1, 1)
        self.tau = tau.reshape(-1, 1)
        self.d0_FTL = d0_FTL.reshape(-1, 1)
        self.gamma = gamma.reshape(-1, 1)
        self.beta = beta.reshape(-1, 1)

        self.dtype = x.dtype
        self.device = x.device
        self.shape = (x.size(0), x.size(1))

    def FTL(self, d_x: torch.Tensor, d_v: torch.Tensor) -> torch.Tensor:
        return d_v / ((d_x / self.d0_FTL) ** (1 + self.gamma))

    def OVM(self, desidered_V: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return (self.tau) ** (-1) * (desidered_V - v)

    def F(
        self,
        model: torch.nn.Module,
        index: torch.Tensor,
        x: torch.Tensor,
        v: torch.Tensor,
        record: bool,
        **kwargs,
    ) -> torch.Tensor:
        d_x = torch.zeros(self.shape[0], dtype=self.dtype, device=self.device)
        d_v = torch.zeros(self.shape[0], dtype=self.dtype, device=self.device)
        mask = index < self.shape[1] - 1
        d_x[mask] = x[mask, index[mask] + 1] - x[mask, index[mask]]
        d_x[~mask] = x[~mask, 0] - x[~mask, -1]
        d_x = torch.remainder(d_x.reshape(-1, 1), self.len_road)
        d_v[mask] = v[mask, index[mask] + 1] - v[mask, index[mask]]
        d_v[~mask] = v[~mask, 0] - v[~mask, -1]
        d_v = d_v.reshape(-1, 1)
        # computo la velocità ideale
        trasformer_mask = kwargs.get(
            "mask", torch.zeros((16, 16), dtype=torch.bool, device=self.device)
        )
        # controllo che self.record sia in __dict__
        if not hasattr(self, "record"):
            raise ValueError("record not found")
        if record:
            # unisco d_x d_v
            new_state = torch.cat([d_x, d_v], dim=1).reshape(-1, 2, 1)
            if hasattr(self, "record"):
                raise ValueError("record not found")
            # devo riportare new_state nel record, quindi shift di 1 di self.record e aggiunta di new_state
            self.record: torch.Tensor = torch.cat(
                [self.record[:, :, 1:], new_state], dim=2
            )
            # computo la velocità ideale
            V = self.Vmax * model(self.record, trasformer_mask)
        else:
            # non è necessario aggiornare il record, si riporta quindi un record fittizio
            new_state = torch.cat([d_x, d_v], dim=1).reshape(-1, 2, 1)
            fake_record = torch.cat([self.record[:, :, 1:], new_state], dim=2)
            V = self.Vmax * model(fake_record, trasformer_mask)
        return (self.beta * self.OVM(V, v) + self.FTL(d_x, d_v)) / (1 + self.beta)

    def auto_F(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        d_x = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        d_v = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        d_x[:, :-1] = x[:, 1:] - x[:, :-1]
        d_x[:, -1] = x[:, 0] - x[:, -1]
        d_x = torch.remainder(d_x, self.len_road)
        d_v[:, :-1] = v[:, 1:] - v[:, :-1]
        d_v[:, -1] = v[:, 0] - v[:, -1]
        # computo la velocità ideale
        V = (
            self.Vmax
            * (
                torch.tanh((d_x - self.d0_OVM) / self.Delta)
                + torch.tanh(self.d0_OVM / self.Delta)
            )
            / (1 + torch.tanh(self.d0_OVM / self.Delta))
        )
        return (self.beta * self.OVM(V, v) + self.FTL(d_x, d_v)) / (1 + self.beta)

    def auto_step(self, dt: float):
        # apply heun method

        # calcolo la forza
        F = self.auto_F(self.x, self.v)
        torch.clamp(F, -max_deceleration, max_acceleration, out=F)

        # f = [v,F]

        d_x = self.v * dt
        d_v = F * dt

        # calcolo nuovamente la forza
        F_heun = self.auto_F(self.x + d_x, self.v + d_v)
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

    def step(self, dt: float, model: torch.nn.Module, index: torch.Tensor):
        # apply heun method

        # calcolo la forza che tutti i veicoli dovrebbero avere
        F = self.auto_F(self.x, self.v)
        F = torch.clamp(F, -max_deceleration, max_acceleration)
        d_x = self.v * dt
        d_v = F * dt
        F_heun = self.auto_F(self.x + d_x, self.v + d_v)
        F_heun = torch.clamp(F_heun, -max_deceleration, max_acceleration)
        d_x = (dt / 2) * (2 * self.v + d_v)
        d_v = (dt / 2) * (F + F_heun)

        # calcolo la forza che suggerisce model sugli agenti indicati da index
        mask = torch.ones((16, 16), dtype=torch.bool, device=self.device)
        mask = torch.triu(mask, diagonal=1)
        F = self.F(model, index, self.x, self.v, False, mask=mask)
        F = torch.clamp(F, -max_deceleration, max_acceleration)
        d_x = self.v * dt
        d_v = F * dt
        F_heun = self.F(model, index, self.x + d_x, self.v + d_v, True, mask=mask)
        F_heun = torch.clamp(F_heun, -max_deceleration, max_acceleration)
        d_x = (dt / 2) * (2 * self.v + d_v)
        d_v = (dt / 2) * (F + F_heun)

        self.x += d_x
        self.v += d_v
        self.x = torch.remainder(self.x, self.len_road)
        self.v = torch.clamp(self.v, torch.zeros_like(self.Vmax), self.Vmax)

        # nan check
        if torch.isnan(self.x).any() or torch.isnan(self.v).any():
            raise ValueError("Nan detected")

    def record_step(self, dt: float, index: torch.Tensor) -> torch.Tensor:
        # apply heun method

        record_d_x = torch.zeros(self.shape[0], dtype=self.dtype, device=self.device)
        record_d_v = torch.zeros(self.shape[0], dtype=self.dtype, device=self.device)
        mask = index < self.shape[1] - 1
        record_d_x[mask] = self.x[mask, index[mask] + 1] - self.x[mask, index[mask]]
        record_d_x[~mask] = self.x[~mask, 0] - self.x[~mask, -1]
        record_d_v[mask] = self.v[mask, index[mask] + 1] - self.v[mask, index[mask]]
        record_d_v[~mask] = self.v[~mask, 0] - self.v[~mask, -1]
        record_d_x = record_d_x.reshape(-1, 1)
        record_d_v = record_d_v.reshape(-1, 1)
        record_d_x = torch.remainder(record_d_x, self.len_road)

        # calcolo la forza
        F = self.auto_F(self.x, self.v)
        torch.clamp(F, -max_deceleration, max_acceleration, out=F)

        # f = [v,F]

        d_x = self.v * dt
        d_v = F * dt

        # calcolo nuovamente la forza
        F_heun = self.auto_F(self.x + d_x, self.v + d_v)
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

        return torch.cat([record_d_x, record_d_v], dim=1)

    def auto_simulate(self, dt: float, n_steps: int):
        for _ in range(n_steps):
            self.auto_step(dt)

    def record_simulate(self, dt: float, index: torch.Tensor, n_steps: int):
        # simulo normalmente per n_steps ma tenendo traccia dei valori letti da index
        self.record = torch.zeros(
            (self.shape[0], 2, n_steps), dtype=self.dtype, device=self.device
        )
        for i in range(n_steps):
            self.record[:, :, i] = self.record_step(dt, index)

    def energy(self) -> torch.Tensor:
        # calcolo l'energia cinetica
        return (0.5 * self.v**2).mean(dim=1) / (self.Vmax**2)

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

    def detach(self):
        self.x = self.x.detach()
        self.v = self.v.detach()
        self.len_road = self.len_road.detach()
        self.d0_OVM = self.d0_OVM.detach()
        self.Delta = self.Delta.detach()
        self.Vmax = self.Vmax.detach()
        self.tau = self.tau.detach()
        self.d0_FTL = self.d0_FTL.detach()
        self.gamma = self.gamma.detach()
        self.beta = self.beta.detach()
        if hasattr(self, "record"):
            self.record = self.record.detach()
        return self

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
