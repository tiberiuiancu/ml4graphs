import time
import numpy as np
import torch
import dgl.sparse as dglsp
import matplotlib.pyplot as plt


def random_adj(shape: tuple[int, int], sparse: float, device=None):
    p_true = 1.0 - sparse
    return torch.rand(shape, device=device) < p_true


def sym_normalize_adj(adj: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute D^-1/2 A D^-1/2 (symmetric normalization).
    Supports dense tensors and PyTorch COO sparse tensors.
    """
    adj = adj.to(torch.float32)
    deg = adj.sum(dim=1)
    d_inv_sqrt = deg.clamp(min=eps).rsqrt()
    return d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]


def to_dgl_sparse(adj: torch.Tensor) -> dglsp.SparseMatrix:
    """
    Convert a (dense or sparse) torch.Tensor adjacency matrix to a DGL SparseMatrix.
    """
    idx = adj.nonzero(as_tuple=False).t()
    val = adj[idx[0], idx[1]]
    shape = adj.shape
    return dglsp.spmatrix(idx, val, shape)


def time_matmul(mat, other_dim: int = 1024):
    shape = [mat.shape[1], other_dim]
    other = torch.rand(shape, dtype=mat.dtype, device=mat.device)
    now = time.time()
    mat @ other
    return time.time() - now


def bench_config(
    shape: tuple[int, int],
    sparse: float,
    device: str = None,
    n_runs: int = 10,
    n_warmup: int = 10,
):
    adj = random_adj(shape, sparse, device)
    adj = sym_normalize_adj(adj)
    sparse_adj = to_dgl_sparse(adj)

    def _time_mat(mat, n_runs):
        return np.mean([time_matmul(mat) for _ in range(n_runs)])

    _time_mat(adj, n_warmup)
    dense = _time_mat(adj, n_runs)

    _time_mat(sparse_adj, n_warmup)
    sparse = _time_mat(sparse_adj, n_runs)

    return dense / sparse


if __name__ == "__main__":
    step = 1024
    end = 2**14
    n_values = list(range(step, end + step, step))
    sp_values = np.linspace(0.95, 0.999, 5)

    heatmap = [[bench_config((n, n), sp, "cuda") for n in n_values] for sp in sp_values]
    heatmap = np.array(heatmap)[::-1]  # flip y axis

    plt.imshow(heatmap, cmap="hot", aspect="auto")
    plt.colorbar(label="dense < sparse")
    plt.xlabel("n (matrix size)")
    plt.ylabel("sp (sparsity)")
    plt.xticks(
        ticks=range(len(n_values)), labels=[str(n) for n in n_values], rotation=45
    )
    plt.yticks(
        ticks=range(len(sp_values)),
        labels=[f"{sp:.2f}" for sp in sp_values[::-1]],
    )
    plt.savefig("img.png")
