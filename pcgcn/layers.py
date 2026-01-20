import torch
from torch import nn

from pcgcn.data import EdgeBlocks


class GraphConvPartLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        adj: EdgeBlocks,
        splits: list[int],
        device="cpu",
        activation: bool = True,
    ):
        super().__init__()

        self.adj = adj
        self.splits = splits
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
        for i in range(len(self.adj)):
            # grab a^l_k
            start_i, end_i = self.splits[i], self.splits[i + 1]

            for j in range(len(self.adj[i])):
                adj_ij = self.adj[i][j]
                if adj_ij is None:
                    continue

                start_j, end_j = self.splits[j], self.splits[j + 1]
                a_j = a[start_j:end_j, :]
                h = adj_ij.to(self.device) @ a_j
                out[start_i:end_i, :] += h

        return self.relu(out) if self.relu is not None else out
