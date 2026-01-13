import dgl
import metis
import networkx as nx
import numpy as np
import torch


def load_dataset(name: str):
    if name == "pubmed":
        return dgl.data.PubmedGraphDataset(raw_dir="data/pubmed")


def partition(graph, k=2):
    """
    Note: while metis supports directly processing an adjacency list, this is incredibly slow.
    Thus we first convert to networkx, then to metis format.
    """
    edges_t = graph.all_edges()
    edges = list(zip(edges_t[0].cpu().tolist(), edges_t[1].cpu().tolist()))
    edges_str = list(map(str, edges))
    nx_g = nx.parse_edgelist(edges_str)
    metis_g = metis.networkx_to_metis(nx_g)
    _, part = metis.part_graph(metis_g, nparts=k)
    return part


def reorder_graph_by_partition(graph, partitions):
    idx = np.argsort(partitions)
    idx_mapping = dict(zip(idx, range(len(idx))))
    map_fun = lambda x: idx_mapping[x]

    edges1, edges2 = graph.all_edges()
    edges1 = edges1.apply_(map_fun)
    edges2 = edges2.apply_(map_fun)

    # create new adj matrix
    adj = g.adj().clone()
    adj.indices = torch.stack([edges1, edges2])

    return adj, idx


def split_adj_matrix(adj: torch.SparseMatrix, partitions):
    k = max(partitions)
    splits = np.cumsum([0] + [partitions.count(x) for x in range(k)])
    return [
        adj[splits[i] : splits[i + 1], splits[j] : splits[j + 1]]
        for j in range(k)
        for i in range(k)
    ]


def load_and_process_dataset(name: str, k: int = None):
    g = load_dataset(name)

    if k is not None:
        part = partition(g, k)
        adj, idx = reorder_graph_by_partition(g, part)
        edge_blocks = split_adj_matrix(adj, part)
    else:
        edge_blocks = [g.adj()]
        idx = list(range(feat.shape[0]))

    feat = g.ndata["feat"][idx]
    train_mask = g.ndata["train_mask"][idx]
    val_mask = g.ndata["val_mask"][idx]
    test_mask = g.ndata["test_mask"][idx]
    label = g.ndata["label"][idx]

    return edge_blocks, feat, train_mask, val_mask, test_mask, label


if __name__ == "__main__":
    g = load_dataset("pubmed")[0]
    print(partition(g, 10))
