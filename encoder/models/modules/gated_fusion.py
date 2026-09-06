import torch
from torch import nn


class GatedFusion(nn.Module):

    def __init__(
        self,
        embed_dim=32,
        hidden_dim=32,
    ):
        super().__init__()

        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, text_feat, image_feat):

        x = torch.cat(
            [text_feat, image_feat],
            dim=-1
        )

        gate = self.gate_network(x)

        fused = (
            gate * text_feat
            + (1.0 - gate) * image_feat
        )

        return fused, gate