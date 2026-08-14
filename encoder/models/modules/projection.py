import torch.nn as nn


class ProjectionMLP(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=None,
        dropout=0.1,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = max(input_dim // 2, output_dim * 2)

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)