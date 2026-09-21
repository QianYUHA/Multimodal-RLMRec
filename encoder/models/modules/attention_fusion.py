import torch
from torch import nn


class AttentionFusion(nn.Module):

    def __init__(
        self,
        embed_dim=32,
        hidden_dim=32,
    ):
        super().__init__()

        # Modality-specific attention transformations
        self.text_attention = nn.Linear(embed_dim, hidden_dim)
        self.image_attention = nn.Linear(embed_dim, hidden_dim)

        # Shared scoring vector
        self.attention_score = nn.Linear(hidden_dim, 1, bias=False)

        self.activation = nn.Tanh()

    def forward(self, text_feat, image_feat):

        # text_feat, image_feat: [num_items, embed_dim]

        text_hidden = self.activation(
            self.text_attention(text_feat)
        )

        image_hidden = self.activation(
            self.image_attention(image_feat)
        )

        # Unnormalized attention scores
        text_score = self.attention_score(text_hidden)
        image_score = self.attention_score(image_hidden)

        # [num_items, 2]
        scores = torch.cat(
            [text_score, image_score],
            dim=-1
        )

        # Normalize across the two modalities
        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        text_weight = attention_weights[:, 0:1]
        image_weight = attention_weights[:, 1:2]

        fused = (
            text_weight * text_feat
            + image_weight * image_feat
        )

        return fused, attention_weights