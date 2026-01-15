from torch import nn
import torch

from pcgcn.data import EdgeBlocks
from pcgcn.layers import GraphConvPartLayer


class PCGCN(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_size: int,
        n_layers: int,
        adj: EdgeBlocks,
        splits: list[int],
        device="cpu",
    ):
        super().__init__()

        self.device = device
        sizes = [in_size] + [hidden_size] * (n_layers)
        layers = [
            GraphConvPartLayer(
                sizes[i],
                sizes[i + 1],
                adj,
                splits,
                device,
            )
            for i in range(n_layers)
        ]

        layers.append(
            nn.Linear(in_features=hidden_size, out_features=out_size, device=device)
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)
