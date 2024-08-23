from abc import ABC, abstractmethod
from typing import List

import torch

from . import simulator


class Transform(ABC):
    @abstractmethod
    def __call__(self, traffic: simulator.databatch) -> simulator.databatch:
        raise NotImplementedError


class Compose(Transform):
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(self, traffic: simulator.databatch) -> simulator.databatch:
        for t in self.transforms:
            traffic = t(traffic)
        return traffic

    def __len__(self) -> int:
        return len(self.transforms)


class RandomNoise(Transform):
    def __init__(self, pos_noise: float, vel_noise: float):
        self.d_x = pos_noise
        self.d_v = vel_noise

    def __call__(self, traffic: simulator.databatch) -> simulator.databatch:
        traffic.x += (
            2 * torch.rand(traffic.shape, dtype=traffic.dtype, device=traffic.device)
            - 1
        ) * self.d_x
        traffic.v *= (
            torch.rand(traffic.shape, dtype=traffic.dtype, device=traffic.device)
            * self.d_v
        )

        # Since the initial data is uniform in both position and velocity, I can reorder it in any way I prefer
        torch.remainder(traffic.x, traffic.len_road, out=traffic.x)
        traffic.x, _ = traffic.x.sort(dim=1)

        return traffic


class RandomDropout(Transform):
    def __init__(self, drop_prob: float):
        self.drop_prob = drop_prob

    def __call__(self, traffic: simulator.databatch) -> simulator.databatch:
        mask = (
            torch.rand(traffic.shape[1], dtype=traffic.x.dtype, device=traffic.x.device)
            > self.drop_prob
        )
        traffic.x = traffic.x[:, mask]
        traffic.v = traffic.v[:, mask]
        traffic.shape = (traffic.shape[0], mask.sum().item())
        return traffic


class Simulation(Transform):
    def __init__(self, dt: float, n_steps: int):
        self.dt = dt
        self.n_steps = n_steps

    def __call__(self, traffic: simulator.databatch) -> simulator.databatch:
        traffic.auto_simulate(self.dt, self.n_steps)
        return traffic
