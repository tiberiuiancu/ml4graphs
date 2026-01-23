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
        preload_adj: int = 0,
    ):
        super().__init__()

        self.adj = adj
        self.splits = splits
        self.device = device
        self.k = len(self.adj)

        self.linear = nn.Linear(
            in_features=in_features, out_features=out_features, device=device
        )
        self.relu = nn.ReLU() if activation else None

        self.is_cuda = next(self.parameters()).is_cuda

        self.custreams = [
            [torch.cuda.Stream() for j in range(self.k) if self.is_cuda]
            for i in range(self.k)
        ]

        self.next_idx = (0, 0)
        self.preloads = {}
        curr = (0, 0)
        self.last_preload = None
        for _ in range(preload_adj):
            self.last_preload = self.preload(curr[0], curr[1])
            curr = self._next_idx(*curr)

    def __len__(self):
        return self.k**2

    def _next_idx(self, i, j):
        j = j + 1
        if j >= len(self.adj):
            i = i + 1
            j = 0
        if i >= len(self.adj):
            i = 0
        return i, j

    def try_async_move(self, i, j):
        """moves adj tensor to cuda via stream, if possible"""
        adj = self.adj[i][j]
        stream = self.custreams[i][j]

        if (
            isinstance(adj, torch.Tensor)
            and self.is_cuda
            and self.last_preload is not None
        ):
            with torch.cuda.stream(stream):
                return (adj.cuda(non_blocking=True), stream)
        return (adj.to(self.device), None)

    def preload(self, i, j):
        """starts preload of tensor at given index"""
        self.preloads[(i, j)] = self.try_async_move(i, j)
        return (i, j)

    def preload_next(self):
        """enqueues a move to cuda of the next tensor in sequence"""
        self.last_preload = self._next_idx(*self.last_preload)
        self.preload(*self.last_preload)

    def retrieve_adj(self, i, j):
        """retrieves adjacency matrix tensor, and triggers preloads"""
        if self.last_preload is None:
            return self.try_async_move(i, j)[0]

        adj, stream = self.preloads[(i, j)]
        if stream is not None:
            stream.synchronize()
        elif adj is not None:
            adj.to(self.device)
        del self.preloads[(i, j)]
        self.preload_next()
        return adj

    def get_next_adj(self):
        """acts as an interator over the adjacency matrices"""
        i, j = self.next_idx[0], self.next_idx[1]
        start_i, end_i = self.splits[i], self.splits[i + 1]
        start_j, end_j = self.splits[j], self.splits[j + 1]
        adj = self.retrieve_adj(i, j)
        self.next_idx = self._next_idx(i, j)
        return adj, start_i, end_i, start_j, end_j

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [N, IN] @ [IN, OUT] = [N, OUT]
        a = self.linear(x)

        # adj @ x: [N, N] @ [N, OUT] = [N, OUT]
        out = torch.zeros_like(a)
        for _ in range(len(self)):
            adj, start_i, end_i, start_j, end_j = self.get_next_adj()
            if adj is None:
                continue

            a_j = a[start_j:end_j, :]
            h = adj @ a_j
            out[start_i:end_i, :] += h

        return self.relu(out) if self.relu is not None else out
