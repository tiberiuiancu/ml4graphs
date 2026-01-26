import torch
from torch import nn

from pcgcn.edge_blocks import EdgeBlocks


class GraphConvPartLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        eb: EdgeBlocks,
        activation: bool = True,
        device="cpu",
    ):
        super().__init__()

        self.eb = eb
        self.device = device

        self.linear = nn.Linear(
            in_features=in_features, out_features=out_features, device=device
        )
        self.relu = nn.ReLU() if activation else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [N, IN] @ [IN, OUT] = [N, OUT]
        a = self.linear(x)

        # adj @ x: [N, N] @ [N, OUT] = [N, OUT]
        out = torch.zeros_like(a)
        for _ in range(len(self.eb)):
            adj, start_i, end_i, start_j, end_j = self.eb.get_next_adj()
            if adj is None:
                continue

            a_j = a[start_j:end_j, :]
            h = adj @ a_j
            out[start_i:end_i, :] += h

        return self.relu(out) if self.relu is not None else out
