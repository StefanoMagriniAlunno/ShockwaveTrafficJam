# source/AC.py
import torch


# input 3 valori
# output 1 valore
class AC(torch.nn.Module):
    def __init__(self):
        super(AC, self).__init__()
        self.sequence1 = torch.nn.Sequential(
            torch.nn.Linear(3, 1),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.tensor):
        x = self.sequence1(x)
        return x

    def load_weights(self, path: str):
        self.load_state_dict(torch.load(path, weights_only=True))

    def save_weights(self, path: str):
        torch.save(self.state_dict(), path)
