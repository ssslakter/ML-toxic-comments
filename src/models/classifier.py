from torch import nn
import torch


class Classifier(nn.Module):
    def __init__(self, input_dim, device: torch.device) -> None:
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 6),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
