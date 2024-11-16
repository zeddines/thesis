import torch
import numpy as np

def BPR(pos, neg):
    loss = torch.sum(torch.log((neg - pos).sigmoid()))
    return loss
