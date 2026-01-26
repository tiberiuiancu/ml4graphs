import time
import os

from pcgcn.data import load_and_process_dataset
from pcgcn.models import PCGCN

import torch
from torch.profiler import ProfilerActivity, profile
import dgl
import networkx as nx


os.makedirs("traces", exist_ok=True)


def gen_graph(n: int, m: int = 100):
    return nx.barabasi_albert_graph(n, m, seed=0)


def _process(g, k, autotune_hidden_size: int = None, preload: int = 0):
    (eb, _, _, _, _, _) = load_and_process_dataset(
        dgl.from_networkx(g),
        k,
        autotune_hidden_size=autotune_hidden_size,
        preload_adj=preload,
        device="cuda",
    )

    return eb


def profile_fwd(
    k: int,
    x: torch.Tensor,
    hidden_size: int = 1024,
    out_size: int = 128,
    preload: int = 0,
):
    n = x.shape[0]
    eb = _process(g, k, autotune_hidden_size=hidden_size, preload=preload)
    model = PCGCN(
        in_size=x.shape[1],
        out_size=out_size,
        hidden_size=hidden_size,
        n_layers=3,
        eb=eb,
        device="cuda",
    )

    _handler = lambda prof: prof.export_chrome_trace(
        f"traces/trace_{n}_{k}_{preload}.json"
    )
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        start = time.time()
        model(x)
        return time.time() - start


if __name__ == "__main__":
    n = 2**14
    m = 100
    k = 2
    feat_size = 512
    hidden_size = 1024
    out_size = 128

    g = gen_graph(n, m)
    print(f"Nodes: {g.number_of_nodes()}, edges: {g.number_of_edges()}")
    x = torch.rand((n, feat_size), device="cuda", dtype=torch.float32)

    profile_fwd(k, x, hidden_size=hidden_size, preload=0)
    profile_fwd(k, x, hidden_size=hidden_size, preload=1)
    profile_fwd(k, x, hidden_size=hidden_size, preload=2)
