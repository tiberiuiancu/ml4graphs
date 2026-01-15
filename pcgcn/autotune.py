import time
import dgl.sparse as dglsp
import numpy as np
import torch


def autotune_sparse(
    sparse: dglsp.SparseMatrix,
    shape: tuple[int, int],
    n_iter: int = 10,
    warmup: int = 10,
):
    def run(mat):
        other = torch.rand(shape, dtype=sparse.dtype, device=sparse.device)
        start = time.time()
        mat @ other
        return time.time() - start

    [run(sparse) for _ in range(warmup)]
    sparse_time = np.mean([run(sparse) for _ in range(n_iter)])

    dense = sparse.to_dense()
    [run(dense) for _ in range(warmup)]
    dense_time = np.mean([run(dense) for _ in range(n_iter)])

    print(sparse_time, dense_time)

    return dense if dense_time < sparse_time else sparse
