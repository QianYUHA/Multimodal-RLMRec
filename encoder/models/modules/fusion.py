import torch
import torch.nn as nn


class FusionMLP(nn.Module):

    def __init__(
        self,
        embed_dim,
        hidden_dim=None,
        dropout=0.1,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = embed_dim * 2

        self.network = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, text_feat, image_feat):

        x = torch.cat(
            [text_feat, image_feat],
            dim=-1
        )

        return self.network(x)