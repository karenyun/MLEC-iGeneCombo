import torch.nn.functional as F
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)

def mse_loss(output, target):
    return F.mse_loss(output, target)

def binary_cross_entropy_with_logits(output, target, pos_weight=3):
    return F.binary_cross_entropy_with_logits(output, target, pos_weight=torch.tensor(pos_weight))