import time
import os

from pcgcn.data import load_and_process_dataset
from pcgcn.edge_blocks import EdgeBlocks
from pcgcn.models import PCGCN

import torch
from torch.profiler import ProfilerActivity, profile
import dgl
import networkx as nx
import argparse


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
    n: int,
    m: int,
    k: int,
    seed: int,
    autotune_hidden_size: int,
    preload: int,
    force_regen: bool,
) -> EdgeBlocks:
    fp = f"cache/{n}_{m}_{k}_{seed}.pt"
    if os.path.exists(fp) and not force_regen:
        return EdgeBlocks.from_file(
            fp,
            device="cuda",
            preload_adj=preload,
        )

    g = gen_graph(n, m, seed)
    eb = _process(g, k, autotune_hidden_size=autotune_hidden_size, preload=preload)
    eb.save(fp)
    return eb


def file_append(fp, txt):
    with open(fp, "a") as f:
        f.write(txt)


def write_results(fp, *args, end=""):
    if not os.path.exists(fp):
        file_append(fp, "n,m,k,seed,n_sparse,val\n")
    file_append(fp, ",".join(list(map(str, list(args)))) + end)


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
    force_regen: bool,
):
    trace_fp = f"out/traces/trace_{n}_{k}_{preload}.json"
    mem_fp = f"out/traces/mem_{n}_{k}_{preload}.pickle"
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
        force_regen=force_regen,
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

    model.train(False)

    def _handler(prof):
        prof.export_chrome_trace(trace_fp)

    torch.cuda.memory._record_memory_history(max_entries=100_000)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            start = time.time()
            model(x)
            elapsed = time.time() - start

    write_results(results_fp, n, m, k, seed, eb.n_sparse(), elapsed, end="\n")
    torch.cuda.memory._dump_snapshot(mem_fp)


def find_min_k(n: int):
    adj_size = n**2 * 4  # bytes
    gpu_memory = (
        torch.cuda.mem_get_info()[0] - 2**30
    )  # gpu mem - 1GB reserved for model

    # assume we have to fit 3 adj in this amount of memory
    mem_per_adj = gpu_memory // 3

    # in how many pieces we have to split the adj to accomodate this
    min_adj = adj_size / mem_per_adj

    # find the next smallest number that is a perfect square
    for k in range(1, 100):
        if k * k >= min_adj:
            return k


if __name__ == "__main__":
    seed = 0
    feat_size = 512
    hidden_size = 1024
    out_size = 128
    preload = 2

    def _launch(n, m, k, force_sparse=False):
        autotune_hidden_size = hidden_size if not force_sparse else None
        return profile_fwd(
            n=n,
            m=m,
            k=k,
            seed=seed,
            hidden_size=hidden_size,
            feat_size=feat_size,
            out_size=out_size,
            preload=preload,
            autotune_hidden_size=autotune_hidden_size,
            force_regen=True,
        )

    parser = argparse.ArgumentParser(description="Profile PCGCN forward pass.")
    parser.add_argument("--n", type=int, required=True, help="Number of nodes")
    parser.add_argument(
        "--m",
        type=int,
        required=True,
        help="Number of edges per node",
    )
    parser.add_argument("--k", type=int, required=True, help="Number of partitions")
    parser.add_argument(
        "--force_sparse", action="store_true", help="Force sparse mode (no autotune)"
    )
    args = parser.parse_args()
    _launch(args.n, args.m, args.k, force_sparse=args.force_sparse)
