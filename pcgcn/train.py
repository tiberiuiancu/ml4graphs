from pcgcn.data import load_and_process_dataset
from pcgcn.models import PCGCN
import torch


def evaluate(
    model: PCGCN,
    feat: torch.Tensor,
    eval_mask: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    with torch.no_grad():
        model.eval()
        logits = model(feat)  # [N, classes]
        correct = logits[eval_mask, :].argmax(dim=-1) == labels[eval_mask]
        return correct.sum() / eval_mask.sum()


def train(
    model: PCGCN,
    iterations: int,
    feat: torch.Tensor,
    train_mask: torch.tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    labels: torch.Tensor,
    do_eval: bool = True,
) -> float:
    opt = torch.optim.Adam(model.parameters())
    loss = torch.nn.CrossEntropyLoss()

    for i in range(iterations):
        model.train()
        opt.zero_grad()
        logits = model(feat)
        iter_loss = loss(logits[train_mask], labels[train_mask])
        iter_loss.backward()
        opt.step()

        if i % 100 == 99 and do_eval:
            print(evaluate(model, feat, val_mask, labels))

    return evaluate(model, feat, test_mask, labels) if do_eval else None


if __name__ == "__main__":
    edge_blocks, feat, train_mask, val_mask, test_mask, labels, splits = (
        load_and_process_dataset("pubmed", 3)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PCGCN(
        in_size=feat.shape[1],
        out_size=labels.max().item() + 1,
        hidden_size=feat.shape[1],
        n_layers=3,
        adj=edge_blocks,
        splits=splits,
        device=device,
    )
    acc = train(
        model=model,
        iterations=1000,
        feat=feat.to(device),
        train_mask=train_mask.to(device),
        val_mask=val_mask.to(device),
        test_mask=test_mask.to(device),
        labels=labels.to(device),
    )

    print(acc)
