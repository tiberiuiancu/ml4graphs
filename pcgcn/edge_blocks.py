import torch
import dgl.sparse as dglsp
import gc

from pcgcn.autotune import autotune


class EdgeBlocks:
    def __init__(
        self,
        eb: list[list[dglsp.SparseMatrix]],
        splits: list[int],
        device="cpu",
        autotune_hidden_size: int = None,
        preload_adj: int = 0,
    ):
        self.eb = eb
        self.splits = splits
        self.device = device
        self.k = len(self.eb)
        self.is_cuda = device != "cpu"

        for i in range(self.k):
            for j in range(self.k):
                if eb[i][j] is not None:
                    if autotune_hidden_size is not None:
                        self.eb[i][j] = autotune(
                            eb[i][j],
                            (eb[i][j].shape[1], autotune_hidden_size),
                        )
                    if isinstance(eb[i][j], dglsp.SparseMatrix):
                        self.eb[i][j] = self.eb[i][j].to(self.device)

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
        if j >= len(self.eb):
            i = i + 1
            j = 0
        if i >= len(self.eb):
            i = 0
        return i, j

    def try_async_move(self, i, j):
        """moves adj tensor to cuda via stream, if possible"""
        adj = self.eb[i][j]
        stream = self.custreams[i][j] if self.is_cuda else None

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

        adj, stream = self.preloads.pop((i, j))
        if stream is not None:
            stream.synchronize()
        elif adj is not None:
            adj = adj.to(self.device)

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

    def n_sparse(self):
        acc = 0
        for i in range(self.k):
            for j in range(self.k):
                if isinstance(self.eb[i][j], dglsp.SparseMatrix):
                    acc += 1
        return acc

    def save(self, fp: str) -> None:
        """saves to file"""

        def _serialize_adj(x):
            if isinstance(x, torch.Tensor):
                o = x
                t = "dense"
            else:
                o = {"val": x.val, "idx": x.indices(), "shape": x.shape}
                t = "sparse"

            return {"type": t, "object": o}

        eb_save = [[_serialize_adj(x) for x in y] for y in self.eb]
        to_save = {"splits": self.splits, "eb": eb_save}
        torch.save(to_save, fp)

    @classmethod
    def from_file(
        cls,
        fp: str,
        device="cpu",
        preload_adj: int = 0,
    ) -> "EdgeBlocks":

        def _deserialize_adj(x):
            o = x["object"]
            if x["type"] == "dense":
                return o
            return dglsp.spmatrix(indices=o["idx"], val=o["val"], shape=o["shape"])

        eb_load = torch.load(fp)
        eb = [[_deserialize_adj(x) for x in y] for y in eb_load["eb"]]
        splits = eb_load["splits"]

        return cls(
            eb=eb,
            splits=splits,
            device=device,
            preload_adj=preload_adj,
            # note: no autotune needed, since presumably this has already been autotuned before
        )
