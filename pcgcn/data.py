from typing import TypeAlias
import dgl
import metis
import networkx as nx
import numpy as np
import dgl.sparse as dglsp
import torch

from pcgcn.autotune import autotune

EdgeBlocks: TypeAlias = list[list[dglsp.SparseMatrix | torch.Tensor]]


def load_dataset(name: str):
    mapping = {
        "pubmed": dgl.data.PubmedGraphDataset,
        "reddit": dgl.data.RedditDataset,
        "cora": dgl.data.CoraFullDataset,
    }

    dataset = mapping.get(name)
    if dataset is None:
        raise ValueError(f"Unknown dataset {name}")

    return dataset(raw_dir=f"data/{name}")[0]


def partition(graph, k=2, ufactor=30):
    """
    Note: while metis supports directly processing an adjacency list, this is incredibly slow.
    Thus we first convert to networkx, then to metis format.
    """
    edges_t = graph.all_edges()
    edges = list(zip(edges_t[0].tolist(), edges_t[1].tolist()))
    stringify = lambda x: f"{x[0]} {x[1]}"
    edges_str = list(map(stringify, edges))
    nx_g = nx.parse_edgelist(edges_str)
    metis_g = metis.networkx_to_metis(nx_g)
    _, part = metis.part_graph(metis_g, nparts=k, ufactor=ufactor)
    return part


def reorder_graph_by_partition(
    adj, partitions, device
) -> tuple[list[dglsp.SparseMatrix], np.array]:
    idx = np.argsort(partitions)
    idx_mapping = dict(zip(idx, range(len(idx))))
    map_fun = lambda x: idx_mapping[x]

    # reorder adj mat
    adj_idx = adj.indices().cpu().apply_(map_fun).to(device)
    adj_val = adj.val

    # calculate k^2 adjacency matrices
    k = max(partitions) + 1
    splits = np.cumsum([0] + [partitions.count(x) for x in range(k)])
    edge_blocks = [[] for _ in range(k)]
    for i in range(k):
        for j in range(k):
            start_i, start_j = splits[i], splits[j]
            end_i, end_j = splits[i + 1], splits[j + 1]

            # select only those elements that are part of the current edge block
            mask_i = (adj_idx[0, :] >= start_i) & (adj_idx[0, :] < end_i)
            mask_j = (adj_idx[1, :] >= start_j) & (adj_idx[1, :] < end_j)
            mask = mask_i & mask_j
            if not mask.any():
                # too sparse lol
                edge_blocks[i].append(None)
                continue

            offsets = torch.tensor(
                [[start_i], [start_j]], dtype=adj_idx.dtype, device=device
            )
            idx_ij = adj_idx[:, mask] - offsets
            val_ij = adj_val[mask]
            adj_ij = dglsp.spmatrix(
                indices=idx_ij,
                val=val_ij,
                shape=(end_i - start_i, end_j - start_j),
            ).to(device)
            edge_blocks[i].append(adj_ij)

    return edge_blocks, idx, splits


def load_and_process_dataset(
    name: str, k: int = 1, ufactor: int = 30, autotune_hidden_size: int = None
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    g = load_dataset(name)
    adj: dglsp.SparseMatrix = g.adj()
    n_nodes = adj.shape[0]

    # add self loops for normalization
    idx = adj.indices()
    val = adj.val
    r = torch.arange(0, n_nodes, 1, dtype=idx.dtype)[None, :]
    o = torch.ones(n_nodes, dtype=val.dtype)
    self_loops = torch.concat((r, r))
    idx = torch.concat((idx, self_loops), dim=-1)
    val = torch.concat((val, o))
    adj = dglsp.spmatrix(indices=idx, val=val).coalesce().to(device)

    # normalize adjacency matrix: D^-1/2 * A * D^-1/2
    deg = dglsp.sum(adj, dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
    D_inv_sqrt = dglsp.diag(deg_inv_sqrt)
    adj = D_inv_sqrt @ adj @ D_inv_sqrt

    if k > 1:
        part = partition(g, k, ufactor)
        edge_blocks, idx, splits = reorder_graph_by_partition(adj, part, device)
    else:
        edge_blocks = [[adj]]
        idx = list(range(n_nodes))
        splits = [0, n_nodes]

    feat = g.ndata["feat"][idx, :]
    try:
        train_mask = g.ndata["train_mask"][idx]
        val_mask = g.ndata["val_mask"][idx]
        test_mask = g.ndata["test_mask"][idx]
    except KeyError:
        # create default masks: first 75% train, next 15% test, last 10% val
        N = len(idx)
        train_end = int(0.75 * N)
        test_end = train_end + int(0.15 * N)
        train_mask = torch.zeros(N, dtype=torch.bool)
        test_mask = torch.zeros(N, dtype=torch.bool)
        val_mask = torch.zeros(N, dtype=torch.bool)
        train_mask[:train_end] = True
        test_mask[train_end:test_end] = True
        val_mask[test_end:] = True

        train_mask = train_mask[idx]
        test_mask = test_mask[idx]
        val_mask = val_mask[idx]

    label = g.ndata["label"][idx]

    if autotune_hidden_size is not None:
        for i in range(k):
            for j in range(k):
                if edge_blocks[i][j] is not None:
                    edge_blocks[i][j] = autotune(
                        edge_blocks[i][j],
                        (edge_blocks[i][j].shape[1], autotune_hidden_size),
                    )

    return edge_blocks, feat, train_mask, val_mask, test_mask, label, splits


if __name__ == "__main__":
    print(
        load_and_process_dataset("cora", k=2, ufactor=1000, autotune_hidden_size=1024)
    )
