from typing import Dict

import torch


def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100


def train_one_epoch(model, dataloader, loss_fn, optimizer, device) -> Dict[str, float]:
    model.train()
    train_loss, train_acc = 0.0, 0.0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return {"loss": train_loss, "acc": train_acc}


def evaluate(model, dataloader, loss_fn, device) -> Dict[str, float]:
    model.eval()
    eval_loss, eval_acc = 0.0, 0.0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            eval_loss += loss_fn(y_pred, y).item()
            eval_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

    eval_loss /= len(dataloader)
    eval_acc /= len(dataloader)
    return {"loss": eval_loss, "acc": eval_acc}
