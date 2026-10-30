"""A small network, trained from random initialization (865 parameters)."""
from torch import nn


class NOxRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        # One real number per row, rather than a one-column matrix.
        return self.network(x).squeeze(-1)
