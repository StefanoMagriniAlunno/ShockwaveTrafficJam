from typing import Tuple

import torch

from . import dataset, simulator, transforms


class DataLoader:
    def __init__(
        self,
        dataset: dataset.Dataset,
        batch_size: int,
        shuffle: bool = False,
        transform: transforms.Compose = transforms.Compose([]),
        **kwargs
    ):
        self.dataset = dataset
        self.batch_size = batch_size

        if dataset.mode == "root":
            self.shuffle = shuffle

            self.transform = transform

            if shuffle:
                self.permutation = torch.randperm(len(dataset))

            # prepare simulation parameters
            self.Vmax_range: Tuple[float, float] = kwargs.get(
                "Vmax", (20.0, 25.0)  # maximum velocity of the vehicles
            )
            self.tau_range: Tuple[float, float] = kwargs.get(
                "tau", (0.5, 2.0)  # reaction time of the vehicles for OVM
            )
            self.d0_FTL_range: Tuple[float, float] = kwargs.get(
                "d0_FTL", (4.0, 5.0)  # minimum distance between vehicles for FTL
            )
            self.gamma_range: Tuple[float, float] = kwargs.get(
                "gamma",
                (
                    1.0,
                    2.0,
                ),  # intensity of the FTL model when the distance between vehicles is less than d0_FTL
            )
            self.beta_range: Tuple[float, float] = kwargs.get(
                "beta", (0.0, 2.0)  # weight for OVM (FTL has weight 1)
            )
            self.n_vehicles: int = kwargs.get(
                "n_vehicles", 100  # max number of vehicles
            )
        else:
            self.shuffle = shuffle
            if len(transform) > 0:
                raise ValueError("Transforms are not allowed in normal mode")
            if shuffle:
                raise ValueError("Shuffle is not allowed in normal mode")

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.dataset.mode == "root":
            # creo i parametri di simulazione per ogni dato
            self.Vmax = (
                torch.rand(
                    len(self.dataset),
                    dtype=self.dataset.dtype,
                    device=self.dataset.device,
                )
                * (self.Vmax_range[1] - self.Vmax_range[0])
                + self.Vmax_range[0]
            )
            self.tau = (
                torch.rand(
                    len(self.dataset),
                    dtype=self.dataset.dtype,
                    device=self.dataset.device,
                )
                * (self.tau_range[1] - self.tau_range[0])
                + self.tau_range[0]
            )
            self.d0_FTL = (
                torch.rand(
                    len(self.dataset),
                    dtype=self.dataset.dtype,
                    device=self.dataset.device,
                )
                * (self.d0_FTL_range[1] - self.d0_FTL_range[0])
                + self.d0_FTL_range[0]
            )
            self.gamma = (
                torch.rand(
                    len(self.dataset),
                    dtype=self.dataset.dtype,
                    device=self.dataset.device,
                )
                * (self.gamma_range[1] - self.gamma_range[0])
                + self.gamma_range[0]
            )
            self.beta = (
                torch.rand(
                    len(self.dataset),
                    dtype=self.dataset.dtype,
                    device=self.dataset.device,
                )
                * (self.beta_range[1] - self.beta_range[0])
                + self.beta_range[0]
            )

        self.current_index: int = 0
        return self

    # l'iteratore deve restituire un batch di dati
    # quindi prende un batch di dati dal dataset, moltiplica positions e velocities per la grandezza del dataset
    # altera i dati con transform
    def __next__(self) -> simulator.databatch:
        if self.current_index >= len(self.dataset):
            raise StopIteration

        batch_size = min(self.batch_size, len(self.dataset) - self.current_index)

        # estraggo il batch di informazioni dal dataset
        if self.shuffle:
            dataset_batch = self.dataset[
                self.permutation[self.current_index : self.current_index + batch_size]
            ]
        else:
            dataset_batch = self.dataset[
                self.current_index : self.current_index + batch_size
            ]

        if self.dataset.mode == "normal":
            # calcolo il numero di veicoli
            n_vehicles = (dataset_batch.size(1) - 8) // 2
            self.current_index += self.batch_size
            return simulator.databatch(
                dataset_batch[:, :n_vehicles],  # x
                dataset_batch[:, n_vehicles : 2 * n_vehicles],  # v
                dataset_batch[:, 2 * n_vehicles : (2 * n_vehicles + 1)],  # len_road
                dataset_batch[:, (2 * n_vehicles + 1) : (2 * n_vehicles + 2)],  # d0_OVM
                dataset_batch[:, (2 * n_vehicles + 2) : (2 * n_vehicles + 3)],  # Delta
                dataset_batch[:, (2 * n_vehicles + 3) : (2 * n_vehicles + 4)],  # Vmax
                dataset_batch[:, (2 * n_vehicles + 4) : (2 * n_vehicles + 5)],  # tau
                dataset_batch[:, (2 * n_vehicles + 5) : (2 * n_vehicles + 6)],  # d0_FTL
                dataset_batch[:, (2 * n_vehicles + 6) : (2 * n_vehicles + 7)],  # gamma
                dataset_batch[:, (2 * n_vehicles + 7) : (2 * n_vehicles + 8)],  # beta
            )

        # con le densità e il numero di veicoli deduco la lunghezza della strada
        len_road = self.n_vehicles / dataset_batch[:, 0]
        # con la lunghezza della strada e il numero di veicoli deduco le posizioni iniziali dei veicoli
        x = (
            torch.arange(
                0, self.n_vehicles, dtype=self.dataset.dtype, device=self.dataset.device
            )
            .unsqueeze(0)
            .repeat(batch_size, 1)
            * len_road.unsqueeze(1)
            / self.n_vehicles
        )
        # con le velocità ottengo il tensore esteso
        v = dataset_batch[:, -1].unsqueeze(1).repeat(1, self.n_vehicles)

        # creo il batch
        batch = simulator.databatch(
            x,
            v,
            len_road,
            dataset_batch[:, 1],
            dataset_batch[:, 2],
            self.Vmax[self.current_index : self.current_index + batch_size],
            self.tau[self.current_index : self.current_index + batch_size],
            self.d0_FTL[self.current_index : self.current_index + batch_size],
            self.gamma[self.current_index : self.current_index + batch_size],
            self.beta[self.current_index : self.current_index + batch_size],
        )

        # applico le trasformazioni
        batch = self.transform(batch)

        self.current_index += self.batch_size
        return batch

    def to(self, device):
        self.dataset.to(device)
        return self

    def reset(self):
        if self.dataset.mode == "root":
            raise ValueError("Reset is not allowed in root mode")
        self.dataset.reset()
        self.__init__(self.dataset, self.batch_size, self.shuffle)
        return self
