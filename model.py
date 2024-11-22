from torch import nn
import numpy as np
from torch.nn.modules import ReLU
import torch


class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(in_dim, in_dim//2),
                                    nn.ReLU(),
                                    nn.Linear(in_dim//2, in_dim//4),
                                    nn.ReLU(),
                                    nn.Linear(in_dim//4, 1)
                                    )
    def forward(self, pos, neg):
        pos_score = self.layers(pos)
        neg_score = self.layers(neg)
        return pos_score, neg_score

    def predict(self, ui):
        score = self.layers(ui)
        return score



