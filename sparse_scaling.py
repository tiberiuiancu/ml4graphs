import time
import networkx as nx
import dgl
import torch
from torch.profiler import profile, ProfilerActivity

from pcgcn.data import load_and_process_dataset
from pcgcn.layers import GraphConvPartLayer


def gen_graph(n: int, m: int = 100):
    return nx.barabasi_albert_graph(n, m, seed=0)


def _process(g, k, autotune_hidden_size: int = None, preload: bool = False):
    edge_blocks, _, _, _, _, _, splits = load_and_process_dataset(
        dgl.from_networkx(g), k, autotune_hidden_size=autotune_hidden_size
    )

    if preload:
        for i in range(len(edge_blocks)):
            for j in range(len(edge_blocks[i])):
                edge_blocks[i][j] = edge_blocks[i][j].to("cuda")
    return edge_blocks, splits


def test_single(
    g, k, x, hidden_dim: int = 1024, autotune: bool = False, preload: bool = False
):
    autotune_hidden_size = hidden_dim if autotune else None
    eb, splits = _process(
        g, k, autotune_hidden_size=autotune_hidden_size, preload=preload
    )
    layer = GraphConvPartLayer(x.shape[1], hidden_dim, eb, splits, device="cuda")
    start = time.time()
    layer(x)
    delta = time.time() - start

    # clean cuda cache to make sure we can fit the next tensor
    # not sure if necessary
    for x in eb:
        for y in x:
            del y
    torch.cuda.empty_cache()

    return delta


def test(n: int, feat_size: int = 512, hidden_dim: int = 1024):
    g = gen_graph(n)
    print(f"Nodes: {g.number_of_nodes()}, edges: {g.number_of_edges()}")
    x = torch.rand((n, feat_size), device="cuda", dtype=torch.float32)

    sparse_time = test_single(
        g, k=1, x=x, hidden_dim=hidden_dim, autotune=False, preload=True
    )
    print(sparse_time)
    dense_time = test_single(
        g, k=2, x=x, hidden_dim=hidden_dim, autotune=True, preload=False
    )
    print(dense_time)


if __name__ == "__main__":
    test(2**13)
