# source/AC.py
import torch


# la rete prende in input un dato seriale a 2 canali di lunghezza N e restituisce un valore tra 0 e 1
# La rete prende in input sequenze temporali a 2 canali lunghe almeno 16000 unità e restituisce un valore tra 0 e 1
class roboCar(torch.nn.Module):
    def __init__(self):
        super(roboCar, self).__init__()

        # l'embedding prende in input dati a 30FPS e restituisce dati con 30/16 FPS
        # l'input è una serie temporale con 2 canali di lunghezza 316 (~10 secondi di valutazione)
        # ritorna una serie temporale con 8 canali di lunghezza 16
        self.embedding = torch.nn.Sequential(
            # section 1: double convolution (316 steps)
            torch.nn.Conv1d(2, 4, 3),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(4, 4, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, 2),
            # section 2: double convolution (156 steps)
            torch.nn.Conv1d(4, 16, 3),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(16, 16, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, 2),
            # section 2: single convolution (76 steps)
            torch.nn.Conv1d(16, 16, 5, bias=False),
            torch.nn.BatchNorm1d(16),
            torch.nn.LeakyReLU(),
            torch.nn.AvgPool1d(2, 2),
            # section 3: single convolution (36 steps)
            torch.nn.Conv1d(16, 16, 5, bias=False),
            torch.nn.BatchNorm1d(16),
            torch.nn.LeakyReLU(),
            torch.nn.AvgPool1d(2, 2),
        )

        # l'encoder prende in input una serie temporale con 16 canali di lunghezza 16
        # e la restituisce con 16 canali di lunghezza 16
        # l'encoder è composto da 2 layer
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=16,
                nhead=1,  # essendoci pochi token non servono tante teste
                dim_feedforward=256,
                batch_first=True,
            ),
            num_layers=2,
            mask_check=True,
            enable_nested_tensor=False,  # mettere False se il numero di teste è dispari
        )

        self.flatten = torch.nn.Flatten()

        # prende in input un array di 128 feature e lo classifica in un valore tra 0 e 1
        self.classifier = torch.nn.Sequential(
            # section 1 : double logical block
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            # section 2 : classification block
            torch.nn.Linear(256, 256, bias=False),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.tensor, mask: torch.tensor) -> torch.tensor:
        x = self.embedding(x)
        if type(x) is not torch.Tensor:
            raise TypeError(f"expected torch.Tensor, got {type(x)}")
        x = x.permute(0, 2, 1)
        x = self.encoder(x, mask=mask)
        x = self.flatten(x)
        x = self.classifier(x)
        if type(x) is not torch.Tensor:
            raise TypeError(f"expected torch.Tensor, got {type(x)}")
        return x

    def load_weights(self, path: str):
        self.load_state_dict(torch.load(path, weights_only=True))

    def save_weights(self, path: str):
        torch.save(self.state_dict(), path)
