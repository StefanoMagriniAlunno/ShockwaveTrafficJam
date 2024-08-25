# source/dataset.py
import os
from typing import Tuple

import torch


# make dataset
def make_traffic(size: int, device: torch.device, path: str, **kwargs):
    """Create a dataset of traffic on a road.

    Args:
        size (int): number of samples in the dataset
        device (torch.device): device that create the dataset
        path (str): path where save the dataset
        **kwargs: other parameters

    Parameters:
        density (float): density of vehicles (vehicles/m)
        d0_OVM (float): minimum distance between veichles
        Delta (Tuple[float, float]): range of the parameter Delta
    """

    # read parameters
    density_range: Tuple[float, float] = kwargs.get(
        "density", (0.05, 0.1)  # range of the density of vehicles
    )
    d0_OVM_range: Tuple[float, float] = kwargs.get(
        "d0_OVM",
        (
            12.0,
            17.0,
        ),  # range of the parameter d0_OVM : security distance between vehicles
    )
    Delta_range: Tuple[float, float] = kwargs.get(
        "Delta",
        (
            0.5,
            1.0,
        ),  # range of the parameter Delta : sensitivity of the driver to the distance between vehicles (Delta=0 -> maximum sensitivity)
    )

    # init parameters
    densities = (
        torch.rand(size, device=device, dtype=torch.float32)
        * (density_range[1] - density_range[0])
        + density_range[0]
    )
    d0_OVM = (
        torch.rand(size, device=device, dtype=torch.float32)
        * (d0_OVM_range[1] - d0_OVM_range[0])
        + d0_OVM_range[0]
    )
    Delta = (
        torch.rand(size, device=device, dtype=torch.float32)
        * (Delta_range[1] - Delta_range[0])
        + Delta_range[0]
    )
    velocities = (
        torch.tanh((densities ** (-1) - d0_OVM) / Delta) + torch.tanh(d0_OVM / Delta)
    ) / (1 + torch.tanh(d0_OVM / Delta))

    dataset = torch.cat(
        [
            densities.unsqueeze(1),
            d0_OVM.unsqueeze(1),
            Delta.unsqueeze(1),
            velocities.unsqueeze(1),
        ],
        dim=1,
    )

    # check if path is a directory and it exists
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a directory")

    used_path = os.path.join(path, "root.pt")
    torch.save(dataset, used_path)


# Dataset class
class Dataset:
    def __init__(self, path: str, root: bool = False):
        self.path = path
        if not os.path.isdir(path):
            raise ValueError(f"{path} is not a directory")

        # load dataset
        if root:
            self.dataset: torch.Tensor = torch.load(
                os.path.join(path, "root.pt"),
                weights_only=False,
                map_location=torch.device("cpu"),
            )
            self.dtype = self.dataset.dtype
            self.device = self.dataset.device
            self.mode = "root"

        else:
            # estraggo i file dalla directory introdotti da ShockwaveTrafficJam
            files = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.split("_")[0] == "ShockwaveTrafficJam"
            ]
            # carico in RAM tutti i file
            self.dataset = [
                torch.load(f, weights_only=False, map_location=torch.device("cpu"))
                for f in files
            ]
            self.dtype = self.dataset[
                0
            ].dtype  # si assume lo stesso dtype per tutti i file
            self.device = torch.device("cpu")  # si sa già che i dati sono in RAM
            self.mode = "normal"

    def __len__(self):
        if type(self.dataset) is list:
            return sum([len(d) for d in self.dataset])
        return len(self.dataset)

    def __getitem__(self, idx) -> torch.Tensor:
        if type(self.dataset) is list:
            if type(idx) is slice:
                idx_start: int | None = idx.start
                idx_stop: int | None = idx.stop

                if idx_start is None:
                    # fa fede stop che deve essere int
                    if idx_stop is None:
                        raise ValueError("Invalid slice")
                    # calcolo l'indice di termine
                    for i, d in enumerate(self.dataset):
                        stop = (i, idx_stop)
                        idx_stop -= len(d)
                        if idx_stop <= 0:
                            break
                    else:
                        raise ValueError("Index out of range")
                    # imposto start automaticamente = (stop[0], 0)
                    start = (stop[0], 0)
                elif idx_stop is None:
                    # fa fede start che sarà per forza int (giunti qui)
                    # calcolo l'indice di partenza
                    for i, d in enumerate(self.dataset):
                        start = (i, idx_start)
                        idx_start -= len(d)
                        if idx_start < 0:
                            break
                    else:
                        raise ValueError("Index out of range")
                    # imposto stop automaticamente = (start[0], len(self.dataset[start[0]]))
                    stop = (start[0], len(self.dataset[start[0]]))
                else:
                    # calcolo l'indice di partenza e di termine
                    for i, d in enumerate(self.dataset):
                        start = (i, idx_start)
                        idx_start -= len(d)
                        if idx_start < 0:
                            break
                    else:
                        raise ValueError("Index out of range")
                    for i, d in enumerate(self.dataset):
                        stop = (i, idx_stop)
                        idx_stop -= len(d)
                        if idx_stop <= 0:
                            break
                    else:
                        raise ValueError("Index out of range")
                    # controllo che i due indici siano dello stesso dataset
                    if start[0] != stop[0]:
                        raise ValueError("Cannot slice multiple datasets")
                return self.dataset[start[0]][start[1] : stop[1] : idx.step]
            # cerco il dataset di appartenenza
            for i, d in enumerate(self.dataset):
                if idx < len(d):
                    return d[idx]
                idx -= len(d)
            raise ValueError("Index out of range")
        return self.dataset[idx]

    def to(self, device: torch.device):
        if type(self.dataset) is list:
            raise ValueError("Cannot move to device a dataset in RAM")
        self.dataset = self.dataset.to(device)
        self.device = device
        return self

    def reset(self):
        if self.mode == "root":
            raise ValueError("Cannot reset a root dataset")
        self.__init__(self.path, root=False)
