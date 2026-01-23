from torch import nn
import torch

from pcgcn.edge_blocks import EdgeBlocks
from pcgcn.layers import GraphConvPartLayer


class PCGCN(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_size: int,
        n_layers: int,
        eb: EdgeBlocks,
        device="cpu",
    ):
        super().__init__()

        self.device = device
        sizes = [in_size] + [hidden_size] * (n_layers)
        layers = [
            GraphConvPartLayer(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                eb=eb,
                device=device,
            )
            for i in range(n_layers)
        ]

        layers.append(
            nn.Linear(in_features=hidden_size, out_features=out_size, device=device)
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)
