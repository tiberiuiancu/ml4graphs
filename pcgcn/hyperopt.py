import hyperopt
from hyperopt import hp
import torch

from pcgcn.train import train
from pcgcn.data import load_and_process_dataset
from pcgcn.models import PCGCN

import time


def _objective(k: int, ufactor: int):
    edge_blocks, feat, train_mask, val_mask, test_mask, labels, splits = (
        load_and_process_dataset("cora", k, ufactor)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PCGCN(
        in_size=feat.shape[1],
        out_size=labels.max().item() + 1,
        hidden_size=1024,
        n_layers=2,
        adj=edge_blocks,
        splits=splits,
        device=device,
    )

    start = time.time()
    train(
        model=model,
        iterations=1,
        feat=feat.to(device),
        train_mask=train_mask.to(device),
        val_mask=val_mask.to(device),
        test_mask=test_mask.to(device),
        labels=labels.to(device),
        do_eval=False,
    )
    return time.time() - start


def hyper_opt(max_evals: int = 100):
    space = hp.choice(
        "args",
        [
            (
                hp.uniformint("k", 2, 5),
                hp.uniformint("ufactor", 30, 100000),
            ),
        ],
    )

    def objective(params):
        loss = _objective(*params)
        return {"loss": loss, "status": hyperopt.STATUS_OK}

    trials = hyperopt.Trials()
    best = hyperopt.fmin(
        fn=objective,
        space=space,
        algo=hyperopt.tpe.suggest,
        max_evals=max_evals,
        trials=trials,
    )

    best_params = {"k": int(best["k"]), "ufactor": int(best["ufactor"])}
    print("best:", best_params)

    print(f"baseline: {objective((1, 30))}")
    return best_params


if __name__ == "__main__":
    hyper_opt(100)
