import torch
from tqdm.notebook import tqdm

from . import AC
from .ShockwaveTrafficJam import dataloader, simulator


def train_sim(
    agent: AC.roboCar,
    batch: simulator.databatch,
    index: torch.Tensor,
    time_step: float,
    steps: int,
    deep_steps: int,
    optimizer: torch.optim.Optimizer,
) -> float:
    if deep_steps <= 1:
        raise ValueError("deep_steps must be greater than 1")
    batch.record_simulate(time_step, index, 316)
    progress_bar = tqdm(range(steps), desc="Simulation", leave=False)
    profit = 0
    for s in progress_bar:
        for _ in range(deep_steps):
            batch.step(time_step, agent, index)
        energy = batch.energy()
        G = energy.mean()
        G.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch = batch.detach()
        G = G.detach()
        # nella barra di progresso viene mostrato il valore dell'energia media G.item()
        profit = G.item()
        progress_bar.set_postfix(
            record=profit, sim_time=(s + 1) * time_step * deep_steps
        )
    return profit


def train(
    agent: AC.roboCar,
    dataloader: dataloader.DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    time_step: float,
    steps: int,
    deep_steps: int,
) -> float:
    progress_bar_epoch = tqdm(range(epochs), desc="Epoch", leave=False)
    epoch_profit: float = 0
    for _ in progress_bar_epoch:
        epoch_profit = 0
        progress_bar_batch = tqdm(dataloader, desc="Batch", leave=False)
        for batch in progress_bar_batch:
            batch.to(torch.device("cuda"))
            idx = torch.randint(
                0, batch.shape[1], (batch.shape[0],), device=torch.device("cuda")
            )
            batch_profit = train_sim(
                agent, batch, idx, time_step, steps, deep_steps, optimizer
            )
            epoch_profit += batch_profit
            progress_bar_batch.set_postfix(
                energy=batch_profit, n_vehicles=batch.shape[1]
            )
        epoch_profit /= len(dataloader)
        progress_bar_epoch.set_postfix(mean_energy=epoch_profit)
    return epoch_profit
