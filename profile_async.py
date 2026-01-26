import time
import os

from pcgcn.data import load_and_process_dataset
from pcgcn.edge_blocks import EdgeBlocks
from pcgcn.models import PCGCN

import torch
from torch.profiler import ProfilerActivity, profile
import dgl
import networkx as nx


os.makedirs("out", exist_ok=True)
os.makedirs("cache", exist_ok=True)
os.makedirs("out/traces", exist_ok=True)


def gen_graph(n: int, m: int = 100, seed: int = 0):
    return nx.barabasi_albert_graph(n, m, seed=seed)


def _process(g, k, autotune_hidden_size: int = None, preload: int = 0) -> EdgeBlocks:
    (eb, _, _, _, _, _) = load_and_process_dataset(
        dgl.from_networkx(g),
        k,
        autotune_hidden_size=autotune_hidden_size,
        preload_adj=preload,
        device="cuda",
    )

    return eb


def load_or_gen_eb(
    n: int, m: int, k: int, seed: int, autotune_hidden_size: int, preload: int
) -> EdgeBlocks:
    fp = f"cache/{n}_{m}_{k}_{seed}.pt"
    if os.path.exists(fp):
        return EdgeBlocks.from_file(
            fp,
            device="cuda",
            autotune_hidden_size=autotune_hidden_size,
            preload_adj=preload,
        )

    g = gen_graph(n, m, seed)
    eb = _process(g, k, autotune_hidden_size=hidden_size, preload=preload)
    eb.save(fp)
    return eb


def write_results(fp, n, m, k, seed, val):
    write_header = not os.path.exists(fp)
    with open(fp, "a") as f:
        if write_header:
            f.write("n,m,k,seed,val\n")
        f.write(f"{n},{m},{k},{seed},{val}\n")


def profile_fwd(
    n: int,
    m: int,
    k: int,
    seed: int,
    hidden_size: int,
    feat_size: int,
    out_size: int,
    preload: int,
    autotune_hidden_size: int,
):
    trace_fp = f"out/traces/trace_{n}_{k}_{preload}.json"
    results_fp = f"out/results.csv"

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    x = torch.rand((n, feat_size), device="cuda", dtype=torch.float32)

    eb = load_or_gen_eb(
        n=n,
        m=m,
        k=k,
        seed=seed,
        autotune_hidden_size=autotune_hidden_size,
        preload=preload,
    )

    n = x.shape[0]
    model = PCGCN(
        in_size=x.shape[1],
        out_size=out_size,
        hidden_size=hidden_size,
        n_layers=3,
        eb=eb,
        device="cuda",
    )

    _handler = lambda prof: prof.export_chrome_trace(trace_fp)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        start = time.time()
        model(x)
        elapsed = time.time() - start

    write_results(results_fp, n, m, k, seed, elapsed)


if __name__ == "__main__":
    n = 2**15
    m = 100
    k = 3
    seed = 0

    feat_size = 512
    hidden_size = 1024
    out_size = 128

    force_sparse = False
    preload = 2
    autotune_hidden_size = hidden_size if not force_sparse else None

    profile_fwd(
        n=n,
        m=m,
        k=k,
        seed=seed,
        hidden_size=hidden_size,
        feat_size=feat_size,
        out_size=out_size,
        preload=preload,
        autotune_hidden_size=autotune_hidden_size,
    )
