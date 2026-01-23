import torch
from pcgcn.layers import GraphConvPartLayer
from sparse_scaling import _process, gen_graph
from torch.profiler import ProfilerActivity, profile


def profile_fwd(eb, splits, x: torch.Tensor, hidden_size: int = 1024, preload: int = 0):
    layer = GraphConvPartLayer(
        x.shape[1], hidden_size, eb, splits, device="cuda", preload_adj=preload
    )

    _handler = lambda prof: prof.export_chrome_trace(f"trace_{preload}.json")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        layer(x)


if __name__ == "__main__":
    n = 2**14
    k = 2
    feat_size = 512
    hidden_size = 1024
    autotune_hidden_size = hidden_size

    g = gen_graph(n)
    print(f"Nodes: {g.number_of_nodes()}, edges: {g.number_of_edges()}")
    x = torch.rand((n, feat_size), device="cuda", dtype=torch.float32)
    eb, splits = _process(
        g, k, autotune_hidden_size=autotune_hidden_size, preload=False
    )

    profile_fwd(eb, splits, x, hidden_size=hidden_size, preload=0)
    profile_fwd(eb, splits, x, hidden_size=hidden_size, preload=1)
    profile_fwd(eb, splits, x, hidden_size=hidden_size, preload=2)
