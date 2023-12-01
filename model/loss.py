'''
loss.py
'''
import torch
from torch import nn

class RMSELoss(nn.Module):

    def __init__(self):
        super(RMSELoss, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, pred, y):
        loss = self.loss_fn(pred, y)
        return torch.sqrt(loss)
