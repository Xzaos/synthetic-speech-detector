"""Small CNN on log-mel features. Four conv blocks, global average pooling
over time (frequency is kept so the head can see where in the spectrum a
difference sits), then a two-layer head that outputs one logit.
"""
import torch
from torch import nn

from .features import N_MELS


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class Detector(nn.Module):
    def __init__(self, widths=(16, 32, 64, 96), hidden=32, n_mels=N_MELS):
        super().__init__()
        chans = (1, *widths)
        self.blocks = nn.Sequential(*[block(a, b) for a, b in zip(chans[:-1], chans[1:])])
        f_out = n_mels // 2 ** len(widths)
        self.head = nn.Sequential(nn.Linear(widths[-1] * f_out, hidden), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, mel):
        h = self.blocks(mel).mean(dim=3)
        return self.head(h.flatten(1)).squeeze(1)


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
