import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score
from transformers import PreTrainedModel


def train_loop(dataloader, model: PreTrainedModel, classifier, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in tqdm(enumerate(dataloader)):
        # Compute prediction and loss
        embed = model(X).last_hidden_state[:, 0]
        pred = classifier(embed)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 512 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, classifier, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, np.zeros(6)

    with torch.no_grad():
        for X, y in tqdm(dataloader):
            embed = model(X).last_hidden_state[:, 0]
            pred = classifier(embed)
            test_loss += loss_fn(pred, y).item()
            correct += (y == logits_to_probability(pred).round()
                        ).cpu().numpy().sum(axis=0)
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {correct}, Avg loss: {test_loss:>8f} \n")


def logits_to_probability(logits):
    return torch.sigmoid(logits)
