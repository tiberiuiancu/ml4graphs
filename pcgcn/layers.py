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
        self.is_cuda = next(self.parameters()).is_cuda
        self.relu = nn.ReLU() if activation else None
        self.stream = torch.cuda.Stream() if self.is_cuda else None

    def try_async_move(self, i, j):
        if isinstance(self.adj[i][j], torch.Tensor) and self.is_cuda:
            with torch.cuda.stream(self.stream):
                return self.adj[i][j].cuda(non_blocking=True)
        elif self.is_cuda and self.adj[i][j] is not None:
            return self.adj[i][j].to(self.device)
        else:
            return self.adj[i][j]

    def preload_next(self, i, j):
        j = j + 1
        if j >= len(self.adj):
            i = i + 1
            j = 0
        if i >= len(self.adj):
            return
        return self.try_async_move(i, j)

    def wait_stream(self):
        if self.stream is not None:
            self.stream.synchronize()

    def cleanup(self):
        if self.is_cuda:
            torch.cuda.empty_cache()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        next_adj = self.try_async_move(0, 0)

        # [N, IN] @ [IN, OUT] = [N, OUT]
        a = self.linear(x)

        # adj @ x: [N, N] @ [N, OUT] = [N, OUT]
        out = torch.zeros_like(a)
        for i in range(len(self.adj)):
            # grab a^l_k
            start_i, end_i = self.splits[i], self.splits[i + 1]

            for j in range(len(self.adj[i])):
                curr_adj = next_adj
                next_adj = self.preload_next(i, j)
                if curr_adj is None:
                    continue

                start_j, end_j = self.splits[j], self.splits[j + 1]
                a_j = a[start_j:end_j, :]

                self.wait_stream()
                h = curr_adj @ a_j
                out[start_i:end_i, :] += h

                # free up memory for new tensor
                del curr_adj
                self.cleanup()

        return self.relu(out) if self.relu is not None else out
